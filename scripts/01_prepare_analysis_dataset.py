from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

EXPERIMENT_PATH = BASE_DIR / "data/private/processed/experiment_results_clean.csv"
SOIL_CHAR_PATH = BASE_DIR / "data/private/processed/soil_characterization_replicates_clean.csv"

SOIL_CHAR_SUMMARY_PATH = BASE_DIR / "data/private/processed/soil_characterization_summary.csv"
ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"


ID_COLUMNS_EXPERIMENT = {
    "sample_code",
    "soil_id",
    "soil_label",
    "p_level_code",
    "p_level_percent",
    "replicate",
    "block",
}

ID_COLUMNS_SOIL_CHAR = {
    "soil_id",
    "soil_label",
    "char_replicate",
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )


def parse_numeric_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "" or text.lower() in {"na", "nan", "none"}:
        return np.nan

    # Brazilian style: 1.234,56
    if "." in text and "," in text:
        text = text.replace(".", "").replace(",", ".")
    # Brazilian decimal comma: 1234,56
    elif "," in text:
        text = text.replace(",", ".")
    # Decimal point: 1234.56 stays unchanged

    return pd.to_numeric(text, errors="coerce")


def convert_numeric_columns(df: pd.DataFrame, id_columns: set[str]) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col not in id_columns:
            out[col] = out[col].map(parse_numeric_value)

    return out


def prepare_experiment(df: pd.DataFrame) -> pd.DataFrame:
    out = convert_numeric_columns(df, ID_COLUMNS_EXPERIMENT)

    # Keep identifiers as clean strings where useful
    out["sample_code"] = out["sample_code"].astype(str).str.strip()
    out["soil_label"] = out["soil_label"].astype(str).str.strip()
    out["block"] = out["block"].astype(str).str.strip()

    # Convert key design columns to integer when possible
    for col in ["soil_id", "p_level_code", "p_level_percent", "replicate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # Mark whether second-cut data exist
    out["cut2_observed"] = np.where(
        out["dry_matter_cut2_g"].notna() & out["P_uptake_cut2_mg_pot"].notna(),
        "yes",
        "no",
    )

    return out


def prepare_soil_characterization(df: pd.DataFrame) -> pd.DataFrame:
    out = convert_numeric_columns(df, ID_COLUMNS_SOIL_CHAR)

    out["soil_label"] = out["soil_label"].astype(str).str.strip()

    for col in ["soil_id", "char_replicate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


def summarize_soil_characterization(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col for col in df.columns
        if col not in {"soil_id", "soil_label", "char_replicate"}
    ]

    summary = (
        df.groupby(["soil_id", "soil_label"], as_index=False)
        .agg(
            n_char_replicates=("char_replicate", "count"),
            **{col: (col, "mean") for col in numeric_cols}
        )
    )

    return summary


def main() -> None:
    experiment_raw = read_csv_flexible(EXPERIMENT_PATH)
    soil_char_raw = read_csv_flexible(SOIL_CHAR_PATH)

    experiment = prepare_experiment(experiment_raw)
    soil_char = prepare_soil_characterization(soil_char_raw)
    soil_summary = summarize_soil_characterization(soil_char)

    analysis = experiment.merge(
        soil_summary,
        on=["soil_id", "soil_label"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_soil_mean"),
    )

    SOIL_CHAR_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    soil_summary.to_csv(SOIL_CHAR_SUMMARY_PATH, index=False)
    analysis.to_csv(ANALYSIS_DATASET_PATH, index=False)

    print("Prepared analysis datasets.")
    print(f"Soil characterization summary: {SOIL_CHAR_SUMMARY_PATH}")
    print(f"Rows: {len(soil_summary)}")
    print(f"Columns: {len(soil_summary.columns)}")

    print(f"\nExperiment analysis dataset: {ANALYSIS_DATASET_PATH}")
    print(f"Rows: {len(analysis)}")
    print(f"Columns: {len(analysis.columns)}")

    missing_soil = analysis[analysis["n_char_replicates"].isna()]
    if missing_soil.empty:
        print("\nOK: all experiment rows matched soil characterization summary.")
    else:
        print("\nWARNING: some experiment rows did not match soil characterization:")
        print(missing_soil[["sample_code", "soil_id", "soil_label"]].to_string(index=False))

    print("\nSecond-cut observed counts:")
    print(analysis["cut2_observed"].value_counts().to_string())


if __name__ == "__main__":
    main()
