from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TABLES_DIR = BASE_DIR / "tables/private"
REPORT_PATH = TABLES_DIR / "private_results_synthesis_report.md"


PATHS = {
    "recovery_summary": TABLES_DIR / "extractor_recovery_rates_summary.csv",
    "predictive_ranking": TABLES_DIR / "extractor_predictive_model_ranking.csv",
    "sensitivity_ranking": TABLES_DIR / "extractor_predictive_model_sensitivity_ranking.csv",
    "slope_tests": TABLES_DIR / "soil_specific_extractor_slope_tests.csv",
    "correlations_by_soil_summary": TABLES_DIR / "extractor_response_correlation_summary_by_response.csv",
}


def read_table(name: str) -> pd.DataFrame:
    path = PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def df_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    formatted = df.copy()

    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            if "p_value" in col or col.endswith("_p") or col == "extractor_p_value":
                formatted[col] = formatted[col].map(
                    lambda x: "" if pd.isna(x) else f"{x:.2e}"
                )
            elif formatted[col].dropna().isin([0, 1]).all() and (
                col.startswith("interaction_supported") or col.startswith("rank_")
            ):
                formatted[col] = formatted[col].map(
                    lambda x: "" if pd.isna(x) else str(int(x))
                )
            else:
                formatted[col] = formatted[col].map(
                    lambda x: "" if pd.isna(x) else f"{x:.{digits}f}"
                )

    return formatted.to_markdown(index=False)


def section_recovery_summary() -> str:
    df = read_table("recovery_summary")

    cols = [
        "extractor",
        "n_soils",
        "median_recovery_rate",
        "mean_recovery_rate",
        "min_recovery_rate",
        "max_recovery_rate",
        "median_r_squared",
    ]

    df = df[cols].sort_values("median_recovery_rate", ascending=False)

    return (
        "## 1. Extractor recovery rates\n\n"
        "Recovery rate is the slope of extracted soil P as a function of applied P dose.\n\n"
        + df_to_markdown(df)
        + "\n\n"
        "**Interpretation note.** Higher recovery does not necessarily mean better biological prediction. "
        "Mehlich-1 recovered more P overall, but predictive performance must be evaluated against plant response.\n"
    )


def section_predictive_models() -> str:
    df = read_table("predictive_ranking")

    primary = df[
        (df["response"].isin(["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"]))
        & (df["model_type"].isin(["soil_fixed_effect", "soil_interaction"]))
    ].copy()

    primary = primary.sort_values(["response", "model_type", "rank_by_aic", "rank_by_rmse"])

    cols = [
        "response",
        "model_type",
        "extractor",
        "n",
        "adj_r_squared",
        "rmse",
        "aic",
        "extractor_p_value",
        "rank_by_aic",
        "rank_by_rmse",
    ]

    return (
        "## 2. Predictive model ranking for plant P uptake\n\n"
        "Models compare extractors as predictors of plant P uptake. "
        "`soil_fixed_effect` controls for soil differences; `soil_interaction` allows extractor slopes to vary among soils.\n\n"
        + df_to_markdown(primary[cols])
        + "\n\n"
        "**Interpretation note.** Mehlich-3 ranked first for total P uptake under both soil-fixed and soil-interaction models. "
        "For first-cut P uptake, Mehlich-3 also ranked first using all observations.\n"
    )


def section_sensitivity() -> str:
    df = read_table("sensitivity_ranking")

    primary = df[
        (df["response"].isin(["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"]))
        & (df["model_type"] == "soil_fixed_effect")
    ].copy()

    primary = primary.sort_values(["response", "dataset", "rank_by_aic", "rank_by_rmse"])

    cols = [
        "response",
        "dataset",
        "extractor",
        "n",
        "n_outliers_removed_for_response",
        "adj_r_squared",
        "rmse",
        "aic",
        "rank_by_aic",
        "rank_by_rmse",
    ]

    return (
        "## 3. Sensitivity to RStudent-flagged observations\n\n"
        "This comparison evaluates whether extractor ranking changes after removing observations flagged as influential for each response.\n\n"
        + df_to_markdown(primary[cols])
        + "\n\n"
        "**Interpretation note.** For total P uptake, Mehlich-3 remained the best-ranked extractor after removing the flagged observation. "
        "For first-cut P uptake, resin slightly surpassed Mehlich-3 after removing flagged observations, but both remained the strongest predictors.\n"
    )


def section_soil_specific_slopes() -> str:
    df = read_table("slope_tests")

    primary = df[df["response"].isin(["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"])].copy()
    primary = primary.sort_values(["response", "delta_aic_interaction_minus_fixed"])

    cols = [
        "response",
        "extractor",
        "n",
        "fixed_adj_r_squared",
        "interaction_adj_r_squared",
        "delta_aic_interaction_minus_fixed",
        "interaction_p_value",
        "interaction_supported_p_lt_0_05",
        "interaction_supported_delta_aic_lt_minus_2",
    ]

    return (
        "## 4. Soil-specific extractor slopes\n\n"
        "This test compares soil-fixed models with soil-interaction models. "
        "A supported interaction indicates that the relationship between extracted P and plant response varies among soils.\n\n"
        + df_to_markdown(primary[cols])
        + "\n\n"
        "**Interpretation note.** Soil-specific slopes were supported for plant P uptake responses across all extractors. "
        "This supports interpretation by soil context rather than relying only on pooled correlations.\n"
    )


def section_within_soil_correlations() -> str:
    df = read_table("correlations_by_soil_summary")

    primary = df[df["response"].isin(["P_uptake_total_mg_pot", "P_uptake_cut1_mg_pot"])].copy()
    primary = primary.sort_values(["response", "median_spearman_rho"], ascending=[True, False])

    cols = [
        "response",
        "extractor",
        "n_soils",
        "median_spearman_rho",
        "mean_spearman_rho",
        "n_positive",
        "n_p_lt_0_05",
    ]

    return (
        "## 5. Within-soil Spearman correlations\n\n"
        "Within-soil correlations reduce the confounding effect of differences in baseline P status among soils.\n\n"
        + df_to_markdown(primary[cols])
        + "\n\n"
        "**Interpretation note.** Olsen and Mehlich-3 had the strongest median within-soil correlations with total P uptake, "
        "while Mehlich-3 ranked highest for first-cut P uptake.\n"
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    report = "\n\n".join([
        "# Private results synthesis\n",
        "This file summarizes private analysis outputs. It should not be committed to GitHub.\n",
        section_recovery_summary(),
        section_predictive_models(),
        section_sensitivity(),
        section_soil_specific_slopes(),
        section_within_soil_correlations(),
        (
            "## Working interpretation\n\n"
            "The current evidence supports using plant P uptake, especially total P uptake, as the primary biological response. "
            "Mehlich-1 showed the greatest overall P recovery and the strongest pooled association, but this appears partly driven by between-soil differences. "
            "When soil effects were controlled, Mehlich-3 was the most consistent predictor of plant P uptake and dry matter responses. "
            "Resin also performed well, especially in sensitivity analysis for first-cut P uptake. "
            "Olsen showed strong within-soil correlations and soil-specific slope behavior, but ranked lower in soil-fixed predictive models for total P uptake. "
            "Critical P levels should be interpreted cautiously and should probably be based on plant P uptake rather than dry matter alone.\n"
        ),
    ])

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Private synthesis report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
