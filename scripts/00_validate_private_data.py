from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

EXPERIMENT_PATH = BASE_DIR / "data/private/processed/experiment_results_clean.csv"
SOIL_CHAR_PATH = BASE_DIR / "data/private/processed/soil_characterization_replicates_clean.csv"


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


def to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def validate_experiment(df: pd.DataFrame) -> None:
    print("\n=== Experiment results ===")

    print(f"Rows: {len(df)}")
    if len(df) == 144:
        print("OK: expected 144 experimental units.")
    else:
        print("WARNING: expected 144 experimental units.")

    required_cols = [
        "sample_code",
        "soil_id",
        "soil_label",
        "p_level_code",
        "p_level_percent",
        "p_dose_mg_dm3",
        "p_applied_mg_pot",
        "replicate",
        "block",
        "P_Mehlich1_mg_dm3",
        "P_Mehlich3_mg_dm3",
        "P_resin_mg_dm3",
        "P_Olsen_mg_dm3",
        "dry_matter_cut1_g",
        "plant_P_cut1_g_kg",
        "P_uptake_cut1_mg_pot",
        "dry_matter_cut2_g",
        "plant_P_cut2_g_kg",
        "P_uptake_cut2_mg_pot",
        "dry_matter_total_g",
        "P_uptake_total_mg_pot",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print("WARNING: missing columns:")
        for col in missing_cols:
            print(f"  - {col}")
        return
    else:
        print("OK: required columns found.")

    duplicated = df["sample_code"].duplicated().sum()
    if duplicated == 0:
        print("OK: sample_code is unique.")
    else:
        print(f"WARNING: duplicated sample_code entries: {duplicated}")

    print("\nCounts by soil_id:")
    print(df["soil_id"].value_counts().sort_index().to_string())

    print("\nCounts by p_level_code:")
    print(df["p_level_code"].value_counts().sort_index().to_string())

    combo_counts = (
        df.groupby(["soil_id", "p_level_code"])
        .size()
        .reset_index(name="n")
    )
    bad_combos = combo_counts[combo_counts["n"] != 4]
    if bad_combos.empty:
        print("OK: each soil × P level combination has 4 rows.")
    else:
        print("WARNING: some soil × P level combinations do not have 4 rows:")
        print(bad_combos.to_string(index=False))

    # Check sample_code logic: soil_id + p_level_code + replicate
    expected_code = (
        df["soil_id"].astype(str).str.strip()
        + df["p_level_code"].astype(str).str.strip()
        + df["replicate"].astype(str).str.strip()
    )
    code_mismatch = df[df["sample_code"].astype(str).str.strip() != expected_code]
    if code_mismatch.empty:
        print("OK: sample_code matches soil_id + p_level_code + replicate.")
    else:
        print("WARNING: sample_code mismatches found:")
        print(code_mismatch[["sample_code", "soil_id", "p_level_code", "replicate"]].to_string(index=False))

    # Numeric checks
    p_dose_mg_dm3 = to_number(df["p_dose_mg_dm3"])
    p_applied_mg_pot = to_number(df["p_applied_mg_pot"])
    expected_p_pot = p_dose_mg_dm3 * 1.7
    dose_diff = (p_applied_mg_pot - expected_p_pot).abs()

    bad_dose = df[dose_diff > 0.05]
    if bad_dose.empty:
        print("OK: p_applied_mg_pot is consistent with p_dose_mg_dm3 × 1.7.")
    else:
        print("WARNING: P dose per pot differs from p_dose_mg_dm3 × 1.7:")
        cols = ["sample_code", "p_dose_mg_dm3", "p_applied_mg_pot"]
        print(bad_dose[cols].to_string(index=False))

    dm1 = to_number(df["dry_matter_cut1_g"])
    dm2 = to_number(df["dry_matter_cut2_g"])
    dmt = to_number(df["dry_matter_total_g"])
    dm_diff = (dmt - (dm1 + dm2)).abs()

    bad_dm_total = df[(dm_diff > 0.05) & dmt.notna() & dm1.notna() & dm2.notna()]
    if bad_dm_total.empty:
        print("OK: dry_matter_total_g is consistent where both cuts are available.")
    else:
        print("WARNING: dry matter total inconsistencies:")
        print(bad_dm_total[["sample_code", "dry_matter_cut1_g", "dry_matter_cut2_g", "dry_matter_total_g"]].to_string(index=False))

    uptake1 = to_number(df["P_uptake_cut1_mg_pot"])
    uptake2 = to_number(df["P_uptake_cut2_mg_pot"])
    uptaket = to_number(df["P_uptake_total_mg_pot"])
    uptake_diff = (uptaket - (uptake1 + uptake2)).abs()

    bad_uptake_total = df[(uptake_diff > 0.05) & uptaket.notna() & uptake1.notna() & uptake2.notna()]
    if bad_uptake_total.empty:
        print("OK: P_uptake_total_mg_pot is consistent where both cuts are available.")
    else:
        print("WARNING: P uptake total inconsistencies:")
        print(bad_uptake_total[["sample_code", "P_uptake_cut1_mg_pot", "P_uptake_cut2_mg_pot", "P_uptake_total_mg_pot"]].to_string(index=False))

    cut2_missing = df[
        (df["dry_matter_cut2_g"].astype(str).str.strip() == "")
        | (df["P_uptake_cut2_mg_pot"].astype(str).str.strip() == "")
    ]
    print(f"\nRows with missing second-cut data: {len(cut2_missing)}")
    if len(cut2_missing) > 0:
        print(cut2_missing[["sample_code", "soil_label", "p_level_code", "replicate", "block"]].to_string(index=False))


def validate_soil_characterization(df: pd.DataFrame) -> None:
    print("\n=== Soil characterization ===")

    print(f"Rows: {len(df)}")
    if len(df) == 18:
        print("OK: expected 18 soil characterization rows.")
    else:
        print("WARNING: expected 18 soil characterization rows.")

    required_cols = [
        "soil_id",
        "soil_label",
        "char_replicate",
        "pH_water",
        "pH_CaCl2",
        "pH_saturation_extract",
        "EC_saturation_extract_dS_m",
        "P_rem_mg_L",
        "P_Mehlich1_initial_mg_dm3",
        "P_Mehlich3_initial_mg_dm3",
        "P_resin_initial_mg_dm3",
        "P_Olsen_initial_mg_dm3",
        "total_sand_g_kg",
        "silt_g_kg",
        "clay_g_kg",
        "bulk_density_g_cm3",
        "particle_density_g_cm3",
        "pot_water_holding_capacity_g_g",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print("WARNING: missing columns:")
        for col in missing_cols:
            print(f"  - {col}")
    else:
        print("OK: key columns found.")

    if {"soil_id", "char_replicate"}.issubset(df.columns):
        duplicated = df[["soil_id", "char_replicate"]].duplicated().sum()
        if duplicated == 0:
            print("OK: soil_id × char_replicate is unique.")
        else:
            print(f"WARNING: duplicated soil_id × char_replicate entries: {duplicated}")

        counts = df.groupby("soil_id").size()
        print("\nCounts by soil_id:")
        print(counts.sort_index().to_string())

        bad_counts = counts[counts != 3]
        if bad_counts.empty:
            print("OK: each soil has 3 characterization replicates.")
        else:
            print("WARNING: some soils do not have 3 characterization replicates:")
            print(bad_counts.to_string())


def main() -> None:
    experiment = read_csv_flexible(EXPERIMENT_PATH)
    soil_char = read_csv_flexible(SOIL_CHAR_PATH)

    validate_experiment(experiment)
    validate_soil_characterization(soil_char)

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
