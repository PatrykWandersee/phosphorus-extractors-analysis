from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"
TABLES_DIR = BASE_DIR / "tables/private"

CORRELATIONS_BY_SOIL_PATH = TABLES_DIR / "extractor_response_correlations_by_soil.csv"
CORRELATION_SUMMARY_PATH = TABLES_DIR / "extractor_response_correlation_summary_by_response.csv"


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


def safe_correlations(subset: pd.DataFrame, x: str, y: str) -> dict:
    clean = subset[[x, y]].dropna()

    if len(clean) < 4:
        return {
            "n": len(clean),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_r, pearson_p = pearsonr(clean[x], clean[y])
    spearman_rho, spearman_p = spearmanr(clean[x], clean[y])

    return {
        "n": len(clean),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def compute_correlations_by_soil(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (soil_id, soil_label), soil_df in df.groupby(["soil_id", "soil_label"]):
        for extractor in EXTRACTOR_COLUMNS:
            for response in RESPONSE_COLUMNS:
                corr = safe_correlations(soil_df, extractor, response)

                rows.append({
                    "soil_id": soil_id,
                    "soil_label": soil_label,
                    "extractor": extractor,
                    "response": response,
                    **corr,
                })

    return pd.DataFrame(rows)


def summarize_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    summary = (
        correlations
        .groupby(["response", "extractor"], as_index=False)
        .agg(
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            min_spearman_rho=("spearman_rho", "min"),
            max_spearman_rho=("spearman_rho", "max"),
            n_soils=("spearman_rho", "count"),
            n_positive=("spearman_rho", lambda x: (x > 0).sum()),
            n_p_lt_0_05=("spearman_p", lambda x: (x < 0.05).sum()),
        )
    )

    return summary.sort_values(
        ["response", "median_spearman_rho"],
        ascending=[True, False],
    )


def main() -> None:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DATASET_PATH)

    correlations = compute_correlations_by_soil(df)
    summary = summarize_correlations(correlations)

    correlations.to_csv(CORRELATIONS_BY_SOIL_PATH, index=False)
    summary.to_csv(CORRELATION_SUMMARY_PATH, index=False)

    print("Correlations by soil finished.")
    print(f"By-soil correlations: {CORRELATIONS_BY_SOIL_PATH}")
    print(f"Summary: {CORRELATION_SUMMARY_PATH}")

    for response in ["P_uptake_total_mg_pot", "dry_matter_total_g", "P_uptake_cut1_mg_pot", "dry_matter_cut1_g"]:
        print(f"\nMedian within-soil Spearman correlations with {response}:")
        top = summary[summary["response"] == response].copy()
        print(
            top[[
                "extractor",
                "n_soils",
                "median_spearman_rho",
                "mean_spearman_rho",
                "n_positive",
                "n_p_lt_0_05",
            ]]
            .sort_values("median_spearman_rho", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
