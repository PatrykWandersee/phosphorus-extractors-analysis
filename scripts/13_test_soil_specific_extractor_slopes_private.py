from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
SLOPE_TESTS_PATH = TABLES_DIR / "soil_specific_extractor_slope_tests.csv"


RESPONSE_COLUMNS = [
    "P_uptake_total_mg_pot",
    "P_uptake_cut1_mg_pot",
    "dry_matter_total_g",
    "dry_matter_cut1_g",
]


def fit_models(data: pd.DataFrame, response: str):
    fixed_formula = f"{response} ~ P_extracted_mg_dm3 + C(soil_label)"
    interaction_formula = f"{response} ~ P_extracted_mg_dm3 * C(soil_label)"

    fixed_model = smf.ols(fixed_formula, data=data).fit()
    interaction_model = smf.ols(interaction_formula, data=data).fit()

    f_stat, p_value, df_diff = interaction_model.compare_f_test(fixed_model)

    return fixed_model, interaction_model, f_stat, p_value, df_diff


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Extractor long-format dataset not found: {EXTRACTOR_LONG_PATH}\n"
            "Run scripts/04_prepare_relative_response_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    rows = []

    for response in RESPONSE_COLUMNS:
        for extractor, group in df.groupby("extractor"):
            data = group[
                [
                    "soil_label",
                    "P_extracted_mg_dm3",
                    response,
                ]
            ].dropna().copy()

            if len(data) < 20:
                continue

            fixed_model, interaction_model, f_stat, p_value, df_diff = fit_models(data, response)

            rows.append({
                "response": response,
                "extractor": extractor,
                "n": int(fixed_model.nobs),
                "fixed_adj_r_squared": fixed_model.rsquared_adj,
                "interaction_adj_r_squared": interaction_model.rsquared_adj,
                "fixed_aic": fixed_model.aic,
                "interaction_aic": interaction_model.aic,
                "delta_aic_interaction_minus_fixed": interaction_model.aic - fixed_model.aic,
                "fixed_rmse": float(np.sqrt(np.mean(fixed_model.resid ** 2))),
                "interaction_rmse": float(np.sqrt(np.mean(interaction_model.resid ** 2))),
                "interaction_f_stat": f_stat,
                "interaction_p_value": p_value,
                "interaction_df_diff": df_diff,
                "interaction_supported_p_lt_0_05": p_value < 0.05,
                "interaction_supported_delta_aic_lt_minus_2": (interaction_model.aic - fixed_model.aic) < -2,
            })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["response", "delta_aic_interaction_minus_fixed"]
    )

    results.to_csv(SLOPE_TESTS_PATH, index=False)

    print("Soil-specific slope tests finished.")
    print(f"Results: {SLOPE_TESTS_PATH}")

    for response in RESPONSE_COLUMNS:
        print(f"\n=== {response} ===")
        subset = results[results["response"] == response].copy()
        print(
            subset[
                [
                    "extractor",
                    "n",
                    "fixed_adj_r_squared",
                    "interaction_adj_r_squared",
                    "delta_aic_interaction_minus_fixed",
                    "interaction_p_value",
                    "interaction_supported_p_lt_0_05",
                    "interaction_supported_delta_aic_lt_minus_2",
                ]
            ]
            .sort_values("delta_aic_interaction_minus_fixed")
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
