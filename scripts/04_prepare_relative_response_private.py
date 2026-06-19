from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"

OUTPUT_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset_with_relative_response.csv"
EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
RELATIVE_RESPONSE_SUMMARY_PATH = TABLES_DIR / "relative_response_summary_by_soil.csv"


EXTRACTOR_MAP = {
    "P_Mehlich1_mg_dm3": "Mehlich-1",
    "P_Mehlich3_mg_dm3": "Mehlich-3",
    "P_resin_mg_dm3": "Resin",
    "P_Olsen_mg_dm3": "Olsen",
}

RESPONSE_COLUMNS = [
    "dry_matter_cut1_g",
    "P_uptake_cut1_mg_pot",
    "dry_matter_cut2_g",
    "P_uptake_cut2_mg_pot",
    "dry_matter_total_g",
    "P_uptake_total_mg_pot",
]


def add_relative_responses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for response in RESPONSE_COLUMNS:
        rel_col = response.replace("_g", "").replace("_mg_pot", "") + "_rel_pct"

        max_by_soil = out.groupby("soil_id")[response].transform("max")
        out[rel_col] = np.where(
            max_by_soil > 0,
            100 * out[response] / max_by_soil,
            np.nan,
        )

    return out


def create_extractor_long_format(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
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

    response_cols = [
        "dry_matter_cut1_g",
        "dry_matter_cut1_rel_pct",
        "P_uptake_cut1_mg_pot",
        "P_uptake_cut1_rel_pct",
        "dry_matter_total_g",
        "dry_matter_total_rel_pct",
        "P_uptake_total_mg_pot",
        "P_uptake_total_rel_pct",
    ]

    available_id_cols = [col for col in id_cols if col in df.columns]
    available_response_cols = [col for col in response_cols if col in df.columns]

    long_df = df.melt(
        id_vars=available_id_cols + available_response_cols,
        value_vars=list(EXTRACTOR_MAP.keys()),
        var_name="extractor_column",
        value_name="P_extracted_mg_dm3",
    )

    long_df["extractor"] = long_df["extractor_column"].map(EXTRACTOR_MAP)
    long_df = long_df.drop(columns=["extractor_column"])

    first_cols = available_id_cols + ["extractor", "P_extracted_mg_dm3"]
    other_cols = [col for col in long_df.columns if col not in first_cols]
    long_df = long_df[first_cols + other_cols]

    return long_df


def summarize_relative_response(df: pd.DataFrame) -> pd.DataFrame:
    rel_cols = [col for col in df.columns if col.endswith("_rel_pct")]

    summary = (
        df.groupby(["soil_id", "soil_label", "p_level_code", "p_level_percent"], as_index=False)
        .agg(
            n=("sample_code", "count"),
            **{f"{col}_mean": (col, "mean") for col in rel_cols},
            **{f"{col}_sd": (col, "std") for col in rel_cols},
        )
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
    df_rel = add_relative_responses(df)

    extractor_long = create_extractor_long_format(df_rel)
    relative_summary = summarize_relative_response(df_rel)

    df_rel.to_csv(OUTPUT_DATASET_PATH, index=False)
    extractor_long.to_csv(EXTRACTOR_LONG_PATH, index=False)
    relative_summary.to_csv(RELATIVE_RESPONSE_SUMMARY_PATH, index=False)

    print("Relative-response preparation finished.")
    print(f"Dataset with relative responses: {OUTPUT_DATASET_PATH}")
    print(f"Rows: {len(df_rel)}")
    print(f"Columns: {len(df_rel.columns)}")

    print(f"\nExtractor long format: {EXTRACTOR_LONG_PATH}")
    print(f"Rows: {len(extractor_long)}")
    print(f"Columns: {len(extractor_long.columns)}")

    print(f"\nRelative response summary: {RELATIVE_RESPONSE_SUMMARY_PATH}")
    print(f"Rows: {len(relative_summary)}")

    print("\nRelative response columns added:")
    for col in [c for c in df_rel.columns if c.endswith('_rel_pct')]:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
