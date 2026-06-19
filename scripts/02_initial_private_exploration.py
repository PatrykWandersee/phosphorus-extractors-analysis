from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"

TABLES_DIR = BASE_DIR / "tables/private"
DATASET_SUMMARY_PATH = TABLES_DIR / "initial_dataset_summary.csv"
MEANS_BY_SOIL_DOSE_PATH = TABLES_DIR / "mean_response_by_soil_p_level.csv"
CORRELATIONS_PATH = TABLES_DIR / "extractor_response_correlations.csv"
MISSING_CUT2_PATH = TABLES_DIR / "missing_second_cut_observations.csv"


EXTRACTOR_COLUMNS = [
    "P_Mehlich1_mg_dm3",
    "P_Mehlich3_mg_dm3",
    "P_resin_mg_dm3",
    "P_Olsen_mg_dm3",
]

RESPONSE_COLUMNS = [
    "dry_matter_cut1_g",
    "P_uptake_cut1_mg_pot",
    "dry_matter_cut2_g",
    "P_uptake_cut2_mg_pot",
    "dry_matter_total_g",
    "P_uptake_total_mg_pot",
]

DESIGN_COLUMNS = [
    "sample_code",
    "soil_id",
    "soil_label",
    "p_level_code",
    "p_level_percent",
    "p_dose_mg_dm3",
    "p_applied_mg_pot",
    "replicate",
    "block",
    "cut2_observed",
]


def load_analysis_dataset() -> pd.DataFrame:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    return pd.read_csv(ANALYSIS_DATASET_PATH)


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"item": "n_rows", "value": len(df)})
    rows.append({"item": "n_soils", "value": df["soil_id"].nunique()})
    rows.append({"item": "n_p_levels", "value": df["p_level_code"].nunique()})
    rows.append({"item": "n_sample_codes", "value": df["sample_code"].nunique()})
    rows.append({"item": "n_cut2_observed_yes", "value": (df["cut2_observed"] == "yes").sum()})
    rows.append({"item": "n_cut2_observed_no", "value": (df["cut2_observed"] == "no").sum()})

    for col in EXTRACTOR_COLUMNS + RESPONSE_COLUMNS:
        rows.append({"item": f"{col}_missing", "value": df[col].isna().sum()})

    return pd.DataFrame(rows)


def summarize_means_by_soil_and_dose(df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "p_dose_mg_dm3",
        "p_applied_mg_pot",
        *EXTRACTOR_COLUMNS,
        *RESPONSE_COLUMNS,
    ]

    grouped = (
        df.groupby(["soil_id", "soil_label", "p_level_code", "p_level_percent"], as_index=False)
        .agg(
            n=("sample_code", "count"),
            **{f"{col}_mean": (col, "mean") for col in summary_cols},
            **{f"{col}_sd": (col, "std") for col in EXTRACTOR_COLUMNS + RESPONSE_COLUMNS},
        )
    )

    return grouped


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for extractor in EXTRACTOR_COLUMNS:
        for response in RESPONSE_COLUMNS:
            subset = df[[extractor, response]].dropna()

            if len(subset) < 3:
                rows.append(
                    {
                        "extractor": extractor,
                        "response": response,
                        "n": len(subset),
                        "pearson_r": np.nan,
                        "pearson_p": np.nan,
                        "spearman_rho": np.nan,
                        "spearman_p": np.nan,
                    }
                )
                continue

            pearson_r, pearson_p = pearsonr(subset[extractor], subset[response])
            spearman_rho, spearman_p = spearmanr(subset[extractor], subset[response])

            rows.append(
                {
                    "extractor": extractor,
                    "response": response,
                    "n": len(subset),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["response", "spearman_rho"], ascending=[True, False])


def export_missing_second_cut(df: pd.DataFrame) -> pd.DataFrame:
    missing = df[df["cut2_observed"] == "no"].copy()
    cols = [
        "sample_code",
        "soil_id",
        "soil_label",
        "p_level_code",
        "p_level_percent",
        "p_dose_mg_dm3",
        "replicate",
        "block",
    ]

    return missing[cols]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_analysis_dataset()

    dataset_summary = summarize_dataset(df)
    means_by_soil_dose = summarize_means_by_soil_and_dose(df)
    correlations = compute_correlations(df)
    missing_cut2 = export_missing_second_cut(df)

    dataset_summary.to_csv(DATASET_SUMMARY_PATH, index=False)
    means_by_soil_dose.to_csv(MEANS_BY_SOIL_DOSE_PATH, index=False)
    correlations.to_csv(CORRELATIONS_PATH, index=False)
    missing_cut2.to_csv(MISSING_CUT2_PATH, index=False)

    print("Initial private exploration finished.")
    print(f"Dataset summary: {DATASET_SUMMARY_PATH}")
    print(f"Mean response by soil and P level: {MEANS_BY_SOIL_DOSE_PATH}")
    print(f"Extractor-response correlations: {CORRELATIONS_PATH}")
    print(f"Missing second-cut observations: {MISSING_CUT2_PATH}")

    print("\nTop Spearman correlations with total P uptake:")
    top = correlations[correlations["response"] == "P_uptake_total_mg_pot"].copy()
    print(
        top[["extractor", "n", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]]
        .sort_values("spearman_rho", ascending=False)
        .to_string(index=False)
    )

    print("\nTop Spearman correlations with total dry matter:")
    top = correlations[correlations["response"] == "dry_matter_total_g"].copy()
    print(
        top[["extractor", "n", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]]
        .sort_values("spearman_rho", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
