from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
MODEL_COMPARISON_PATH = TABLES_DIR / "extractor_predictive_model_comparison.csv"
MODEL_RANKING_PATH = TABLES_DIR / "extractor_predictive_model_ranking.csv"


RESPONSE_COLUMNS = [
    "P_uptake_total_mg_pot",
    "P_uptake_cut1_mg_pot",
    "dry_matter_total_g",
    "dry_matter_cut1_g",
]

MODEL_FORMULAS = {
    "pooled_linear": "{response} ~ P_extracted_mg_dm3",
    "soil_fixed_effect": "{response} ~ P_extracted_mg_dm3 + C(soil_label)",
    "soil_interaction": "{response} ~ P_extracted_mg_dm3 * C(soil_label)",
}


def rmse(observed, predicted) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def fit_model(data: pd.DataFrame, response: str, model_name: str, formula_template: str):
    formula = formula_template.format(response=response)
    model = smf.ols(formula=formula, data=data).fit()

    predicted = model.predict(data)
    observed = data[response]

    return {
        "response": response,
        "model_type": model_name,
        "formula": formula,
        "n": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "rmse": rmse(observed, predicted),
        "extractor_slope": model.params.get("P_extracted_mg_dm3", np.nan),
        "extractor_p_value": model.pvalues.get("P_extracted_mg_dm3", np.nan),
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
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

            if len(data) < 12:
                continue

            for model_name, formula_template in MODEL_FORMULAS.items():
                try:
                    result = fit_model(data, response, model_name, formula_template)
                    result["extractor"] = extractor
                    rows.append(result)
                except Exception as error:
                    rows.append({
                        "response": response,
                        "extractor": extractor,
                        "model_type": model_name,
                        "formula": formula_template.format(response=response),
                        "n": len(data),
                        "r_squared": np.nan,
                        "adj_r_squared": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "rmse": np.nan,
                        "extractor_slope": np.nan,
                        "extractor_p_value": np.nan,
                        "error": str(error),
                    })

    return pd.DataFrame(rows)


def rank_models(comparison: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison.copy()

    ranked["rank_by_aic"] = (
        ranked.groupby(["response", "model_type"])["aic"]
        .rank(method="min", ascending=True)
    )

    ranked["rank_by_rmse"] = (
        ranked.groupby(["response", "model_type"])["rmse"]
        .rank(method="min", ascending=True)
    )

    ranked["rank_by_adj_r_squared"] = (
        ranked.groupby(["response", "model_type"])["adj_r_squared"]
        .rank(method="min", ascending=False)
    )

    ranked = ranked.sort_values(
        ["response", "model_type", "rank_by_aic", "rank_by_rmse"]
    )

    return ranked


def print_primary_rankings(ranking: pd.DataFrame) -> None:
    for response in RESPONSE_COLUMNS:
        print(f"\n=== {response} ===")

        for model_type in ["pooled_linear", "soil_fixed_effect", "soil_interaction"]:
            subset = ranking[
                (ranking["response"] == response)
                & (ranking["model_type"] == model_type)
            ].copy()

            if subset.empty:
                continue

            print(f"\nModel: {model_type}")
            print(
                subset[
                    [
                        "extractor",
                        "n",
                        "adj_r_squared",
                        "rmse",
                        "aic",
                        "extractor_slope",
                        "extractor_p_value",
                        "rank_by_aic",
                        "rank_by_rmse",
                    ]
                ]
                .sort_values(["rank_by_aic", "rank_by_rmse"])
                .to_string(index=False)
            )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Extractor long-format dataset not found: {EXTRACTOR_LONG_PATH}\n"
            "Run scripts/04_prepare_relative_response_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    comparison = compare_models(df)
    ranking = rank_models(comparison)

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    ranking.to_csv(MODEL_RANKING_PATH, index=False)

    print("Extractor predictive model comparison finished.")
    print(f"Model comparison: {MODEL_COMPARISON_PATH}")
    print(f"Model ranking: {MODEL_RANKING_PATH}")

    print_primary_rankings(ranking)


if __name__ == "__main__":
    main()
