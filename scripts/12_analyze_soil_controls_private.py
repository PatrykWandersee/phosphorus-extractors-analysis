from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]

SOIL_SUMMARY_PATH = BASE_DIR / "data/private/processed/soil_characterization_summary.csv"
RECOVERY_RATES_PATH = BASE_DIR / "tables/private/extractor_recovery_rates_by_soil.csv"
CORRELATIONS_BY_SOIL_PATH = BASE_DIR / "tables/private/extractor_response_correlations_by_soil.csv"

TABLES_DIR = BASE_DIR / "tables/private"
RECOVERY_SOIL_CORR_PATH = TABLES_DIR / "recovery_rate_soil_attribute_correlations.csv"
PERFORMANCE_SOIL_CORR_PATH = TABLES_DIR / "extractor_performance_soil_attribute_correlations.csv"


SOIL_ATTRIBUTES = [
    "pH_water",
    "pH_CaCl2",
    "pH_saturation_extract",
    "EC_saturation_extract_dS_m",
    "P_saturation_extract_mg_L",
    "Ca_saturation_extract_mmolc_L",
    "Mg_saturation_extract_mmolc_L",
    "Na_saturation_extract_mmolc_L",
    "K_saturation_extract_mmolc_L",
    "SAR",
    "Ca_exchangeable_cmolc_dm3",
    "Mg_exchangeable_cmolc_dm3",
    "Na_exchangeable_cmolc_dm3",
    "K_exchangeable_cmolc_dm3",
    "CEC_cmolc_dm3",
    "ESP_pct",
    "TOC_dag_kg",
    "P_rem_mg_L",
    "P_Mehlich1_initial_mg_dm3",
    "P_Mehlich3_initial_mg_dm3",
    "P_resin_initial_mg_dm3",
    "P_Olsen_initial_mg_dm3",
    "total_sand_g_kg",
    "silt_g_kg",
    "clay_g_kg",
    "water_dispersible_clay_g_kg",
    "flocculation_index",
    "bulk_density_g_cm3",
    "particle_density_g_cm3",
    "total_porosity_m3_m3",
    "pot_water_holding_capacity_g_g",
]


def safe_corr(x: pd.Series, y: pd.Series) -> dict:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(clean) < 4:
        return {
            "n": len(clean),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    if clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return {
            "n": len(clean),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_r, pearson_p = pearsonr(clean["x"], clean["y"])
    spearman_rho, spearman_p = spearmanr(clean["x"], clean["y"])

    return {
        "n": len(clean),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def available_soil_attributes(soil: pd.DataFrame) -> list[str]:
    return [col for col in SOIL_ATTRIBUTES if col in soil.columns]


def analyze_recovery_controls(soil: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    merged = recovery.merge(
        soil,
        on=["soil_id", "soil_label"],
        how="left",
        validate="many_to_one",
    )

    rows = []
    attrs = available_soil_attributes(soil)

    for extractor, group in merged.groupby("extractor"):
        for attr in attrs:
            corr = safe_corr(group[attr], group["recovery_rate_slope"])

            rows.append({
                "analysis": "recovery_rate_vs_soil_attribute",
                "extractor": extractor,
                "soil_attribute": attr,
                "target": "recovery_rate_slope",
                **corr,
            })

    out = pd.DataFrame(rows)
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    return out.sort_values(
        ["extractor", "abs_spearman_rho"],
        ascending=[True, False],
    )


def analyze_performance_controls(soil: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    selected_responses = [
        "P_uptake_total_mg_pot",
        "P_uptake_cut1_mg_pot",
        "dry_matter_total_g",
        "dry_matter_cut1_g",
    ]

    performance = performance[
        performance["response"].isin(selected_responses)
    ].copy()

    merged = performance.merge(
        soil,
        on=["soil_id", "soil_label"],
        how="left",
        validate="many_to_one",
    )

    rows = []
    attrs = available_soil_attributes(soil)

    for (extractor, response), group in merged.groupby(["extractor", "response"]):
        for attr in attrs:
            corr = safe_corr(group[attr], group["spearman_rho"])

            rows.append({
                "analysis": "within_soil_spearman_vs_soil_attribute",
                "extractor": extractor,
                "response": response,
                "soil_attribute": attr,
                "target": "within_soil_spearman_rho",
                **corr,
            })

    out = pd.DataFrame(rows)
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    return out.sort_values(
        ["response", "extractor", "abs_spearman_rho"],
        ascending=[True, True, False],
    )


def print_top_recovery_controls(df: pd.DataFrame) -> None:
    print("\nTop soil controls of recovery rate by extractor:")
    for extractor, group in df.groupby("extractor"):
        print(f"\n{extractor}")
        print(
            group[
                [
                    "soil_attribute",
                    "n",
                    "spearman_rho",
                    "spearman_p",
                    "pearson_r",
                    "pearson_p",
                ]
            ]
            .head(8)
            .to_string(index=False)
        )


def print_top_performance_controls(df: pd.DataFrame) -> None:
    print("\nTop soil controls of within-soil extractor performance:")
    for response in ["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"]:
        print(f"\nResponse: {response}")

        subset = df[df["response"] == response].copy()

        for extractor, group in subset.groupby("extractor"):
            print(f"\n{extractor}")
            print(
                group[
                    [
                        "soil_attribute",
                        "n",
                        "spearman_rho",
                        "spearman_p",
                        "pearson_r",
                        "pearson_p",
                    ]
                ]
                .head(6)
                .to_string(index=False)
            )


def main() -> None:
    if not SOIL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Soil characterization summary not found: {SOIL_SUMMARY_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    if not RECOVERY_RATES_PATH.exists():
        raise FileNotFoundError(
            f"Recovery rates not found: {RECOVERY_RATES_PATH}\n"
            "Run scripts/06_estimate_extractor_recovery_rates_private.py first."
        )

    if not CORRELATIONS_BY_SOIL_PATH.exists():
        raise FileNotFoundError(
            f"By-soil correlations not found: {CORRELATIONS_BY_SOIL_PATH}\n"
            "Run scripts/03_correlations_by_soil_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    soil = pd.read_csv(SOIL_SUMMARY_PATH)
    recovery = pd.read_csv(RECOVERY_RATES_PATH)
    performance = pd.read_csv(CORRELATIONS_BY_SOIL_PATH)

    recovery_controls = analyze_recovery_controls(soil, recovery)
    performance_controls = analyze_performance_controls(soil, performance)

    recovery_controls.to_csv(RECOVERY_SOIL_CORR_PATH, index=False)
    performance_controls.to_csv(PERFORMANCE_SOIL_CORR_PATH, index=False)

    print("Soil-control analysis finished.")
    print(f"Recovery-rate soil controls: {RECOVERY_SOIL_CORR_PATH}")
    print(f"Extractor-performance soil controls: {PERFORMANCE_SOIL_CORR_PATH}")

    print_top_recovery_controls(recovery_controls)
    print_top_performance_controls(performance_controls)

    print("\nNote: correlations are exploratory because n = 6 soils.")


if __name__ == "__main__":
    main()
