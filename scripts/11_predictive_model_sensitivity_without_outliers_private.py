from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"
OUTLIERS_PATH = BASE_DIR / "tables/private/plant_response_rstudent_outliers.csv"

TABLES_DIR = BASE_DIR / "tables/private"
SENSITIVITY_PATH = TABLES_DIR / "extractor_predictive_model_sensitivity_without_outliers.csv"
SENSITIVITY_RANKING_PATH = TABLES_DIR / "extractor_predictive_model_sensitivity_ranking.csv"


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


def get_outlier_sample_codes(outliers: pd.DataFrame, response: str) -> set[str]:
    if outliers.empty:
        return set()

    subset = outliers[outliers["response"] == response].copy()
    return set(subset["sample_code"].astype(str))


def compare_models(df: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for response in RESPONSE_COLUMNS:
        response_outliers = get_outlier_sample_codes(outliers, response)

        datasets = {
            "all_data": df,
            "without_response_rstudent_outliers": df[
                ~df["sample_code"].astype(str).isin(response_outliers)
            ].copy(),
        }

        for dataset_name, dataset in datasets.items():
            for extractor, group in dataset.groupby("extractor"):
                data = group[
                    [
                        "sample_code",
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
                        result["dataset"] = dataset_name
                        result["extractor"] = extractor
                        result["n_outliers_removed_for_response"] = len(response_outliers)
                        rows.append(result)
                    except Exception as error:
                        rows.append({
                            "dataset": dataset_name,
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
                            "n_outliers_removed_for_response": len(response_outliers),
                            "error": str(error),
                        })

    return pd.DataFrame(rows)


def rank_models(comparison: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison.copy()

    ranked["rank_by_aic"] = (
        ranked.groupby(["dataset", "response", "model_type"])["aic"]
        .rank(method="min", ascending=True)
    )

    ranked["rank_by_rmse"] = (
        ranked.groupby(["dataset", "response", "model_type"])["rmse"]
        .rank(method="min", ascending=True)
    )

    ranked["rank_by_adj_r_squared"] = (
        ranked.groupby(["dataset", "response", "model_type"])["adj_r_squared"]
        .rank(method="min", ascending=False)
    )

    return ranked.sort_values(
        ["dataset", "response", "model_type", "rank_by_aic", "rank_by_rmse"]
    )


def print_primary_comparison(ranking: pd.DataFrame) -> None:
    for response in ["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"]:
        print(f"\n=== {response} | soil_fixed_effect ===")
        subset = ranking[
            (ranking["response"] == response)
            & (ranking["model_type"] == "soil_fixed_effect")
        ].copy()

        print(
            subset[
                [
                    "dataset",
                    "extractor",
                    "n",
                    "n_outliers_removed_for_response",
                    "adj_r_squared",
                    "rmse",
                    "aic",
                    "extractor_p_value",
                    "rank_by_aic",
                    "rank_by_rmse",
                ]
            ]
            .sort_values(["dataset", "rank_by_aic", "rank_by_rmse"])
            .to_string(index=False)
        )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Extractor long-format dataset not found: {EXTRACTOR_LONG_PATH}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)
    df["sample_code"] = df["sample_code"].astype(str)

    if OUTLIERS_PATH.exists() and OUTLIERS_PATH.stat().st_size > 0:
        outliers = pd.read_csv(OUTLIERS_PATH)
        outliers["sample_code"] = outliers["sample_code"].astype(str)
    else:
        outliers = pd.DataFrame()

    comparison = compare_models(df, outliers)
    ranking = rank_models(comparison)

    comparison.to_csv(SENSITIVITY_PATH, index=False)
    ranking.to_csv(SENSITIVITY_RANKING_PATH, index=False)

    print("Predictive model sensitivity analysis finished.")
    print(f"Sensitivity results: {SENSITIVITY_PATH}")
    print(f"Sensitivity ranking: {SENSITIVITY_RANKING_PATH}")

    print_primary_comparison(ranking)


if __name__ == "__main__":
    main()
