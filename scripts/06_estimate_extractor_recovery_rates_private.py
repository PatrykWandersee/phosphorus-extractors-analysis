from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"

TABLES_DIR = BASE_DIR / "tables/private"
RECOVERY_RATES_PATH = TABLES_DIR / "extractor_recovery_rates_by_soil.csv"
RECOVERY_SUMMARY_PATH = TABLES_DIR / "extractor_recovery_rates_summary.csv"


EXTRACTOR_COLUMNS = {
    "P_Mehlich1_mg_dm3": "Mehlich-1",
    "P_Mehlich3_mg_dm3": "Mehlich-3",
    "P_resin_mg_dm3": "Resin",
    "P_Olsen_mg_dm3": "Olsen",
}


def fit_linear_recovery(group: pd.DataFrame, extractor_col: str) -> dict:
    clean = group[["p_dose_mg_dm3", extractor_col]].dropna().copy()

    if len(clean) < 3:
        return {
            "n": len(clean),
            "intercept": np.nan,
            "recovery_rate_slope": np.nan,
            "r_value": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "std_error": np.nan,
        }

    result = linregress(clean["p_dose_mg_dm3"], clean[extractor_col])

    return {
        "n": len(clean),
        "intercept": result.intercept,
        "recovery_rate_slope": result.slope,
        "r_value": result.rvalue,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "std_error": result.stderr,
    }


def estimate_recovery_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (soil_id, soil_label), group in df.groupby(["soil_id", "soil_label"]):
        for extractor_col, extractor_name in EXTRACTOR_COLUMNS.items():
            fit = fit_linear_recovery(group, extractor_col)

            rows.append({
                "soil_id": soil_id,
                "soil_label": soil_label,
                "extractor": extractor_name,
                "extractor_column": extractor_col,
                **fit,
            })

    return pd.DataFrame(rows)


def summarize_recovery_rates(recovery: pd.DataFrame) -> pd.DataFrame:
    summary = (
        recovery
        .groupby("extractor", as_index=False)
        .agg(
            n_soils=("soil_id", "count"),
            median_recovery_rate=("recovery_rate_slope", "median"),
            mean_recovery_rate=("recovery_rate_slope", "mean"),
            min_recovery_rate=("recovery_rate_slope", "min"),
            max_recovery_rate=("recovery_rate_slope", "max"),
            median_r_squared=("r_squared", "median"),
            mean_r_squared=("r_squared", "mean"),
        )
        .sort_values("median_recovery_rate", ascending=False)
    )

    return summary


def main() -> None:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DATASET_PATH)

    recovery = estimate_recovery_rates(df)
    summary = summarize_recovery_rates(recovery)

    recovery.to_csv(RECOVERY_RATES_PATH, index=False)
    summary.to_csv(RECOVERY_SUMMARY_PATH, index=False)

    print("Extractor recovery-rate estimation finished.")
    print(f"Recovery rates by soil: {RECOVERY_RATES_PATH}")
    print(f"Summary: {RECOVERY_SUMMARY_PATH}")

    print("\nRecovery-rate summary:")
    print(
        summary[[
            "extractor",
            "n_soils",
            "median_recovery_rate",
            "mean_recovery_rate",
            "min_recovery_rate",
            "max_recovery_rate",
            "median_r_squared",
        ]].to_string(index=False)
    )

    print("\nRecovery rates by soil:")
    print(
        recovery[[
            "soil_label",
            "extractor",
            "recovery_rate_slope",
            "r_squared",
            "p_value",
        ]]
        .sort_values(["soil_label", "recovery_rate_slope"], ascending=[True, False])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
