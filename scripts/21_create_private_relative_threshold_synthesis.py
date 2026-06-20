from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

TABLES_DIR = BASE_DIR / "tables/private"
REPORT_DIR = BASE_DIR / "manuscript/private"
REPORT_PATH = REPORT_DIR / "relative_response_threshold_synthesis_private.md"


GLOBAL_THRESHOLDS = TABLES_DIR / "relative_response_global_thresholds.csv"
BY_SOIL_SUMMARY = TABLES_DIR / "relative_response_threshold_summary_by_soil.csv"
LOGISTIC_MODELS = TABLES_DIR / "relative_response_logistic_model_results.csv"
PROBABILITY_RANGE_CHECK = TABLES_DIR / "relative_response_probability_threshold_range_check.csv"
BOOTSTRAP_SUMMARY = TABLES_DIR / "relative_response_threshold_bootstrap_summary.csv"
LOSO_SUMMARY = TABLES_DIR / "relative_response_threshold_leave_one_soil_out_summary.csv"
OUTLIER_SENSITIVITY = TABLES_DIR / "relative_response_threshold_sensitivity_summary_without_outliers.csv"


PRIMARY_RESPONSES = [
    "P_uptake_total_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_cut1_rel_pct",
]

RESPONSE_LABELS = {
    "P_uptake_total_rel_pct": "Total relative P uptake",
    "dry_matter_total_rel_pct": "Total relative dry matter",
    "P_uptake_cut1_rel_pct": "First-cut relative P uptake",
    "dry_matter_cut1_rel_pct": "First-cut relative dry matter",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path)


def fmt_float(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def fmt_table(df: pd.DataFrame, columns: list[str], sort_by=None, ascending=True) -> str:
    work = df.copy()

    if sort_by is not None:
        work = work.sort_values(sort_by, ascending=ascending)

    return work[columns].to_markdown(index=False)


def add_section(lines: list[str], title: str) -> None:
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    global_thresholds = read_csv(GLOBAL_THRESHOLDS)
    by_soil = read_csv(BY_SOIL_SUMMARY)
    logistic = read_csv(LOGISTIC_MODELS)
    range_check = read_csv(PROBABILITY_RANGE_CHECK)
    bootstrap = read_csv(BOOTSTRAP_SUMMARY)
    loso = read_csv(LOSO_SUMMARY)
    sensitivity = read_csv(OUTLIER_SENSITIVITY)

    lines = []

    lines.append("# Private synthesis: relative response thresholds for phosphorus extractors")
    lines.append("")
    lines.append("This private note summarizes the relative-response threshold analyses performed for the phosphorus extractor dataset. It is intended as an internal interpretation document and should not be made public with the original data or numerical outputs.")
    lines.append("")
    lines.append("Analyses covered diagnostic thresholds based on balanced accuracy, probabilistic logistic thresholds, observed-range checks, bootstrap stability, leave-one-soil-out validation, and response-specific outlier sensitivity.")
    lines.append("")

    add_section(lines, "Main interpretation")

    lines.append("The most consistent result is that total relative P uptake is a stronger biological basis for diagnostic P thresholds than relative dry matter. Across the analyses, dry matter-based thresholds showed weaker classification performance and weaker model support, while P uptake-based thresholds were more informative.")
    lines.append("")
    lines.append("However, the leave-one-soil-out validation showed limited transferability of a single global threshold across soils. Therefore, the current dataset supports a methodological interpretation rather than a definitive universal critical P value.")
    lines.append("")
    lines.append("The safest interpretation is:")
    lines.append("")
    lines.append("> Relative P uptake improves diagnostic threshold performance compared with dry matter, but threshold transferability among alkaline soils is limited. This supports soil-specific calibration or at least explicit soil-effect control rather than a single universal threshold.")
    lines.append("")

    add_section(lines, "1. Global diagnostic thresholds at 90% adequacy")

    g90 = global_thresholds[
        (global_thresholds["adequacy_cutoff_pct"] == 90.0)
        & (global_thresholds["response"].isin(PRIMARY_RESPONSES))
    ].copy()

    cols = [
        "response",
        "extractor",
        "n",
        "n_adequate",
        "n_inadequate",
        "threshold_mg_dm3",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]

    lines.append(fmt_table(
        g90,
        cols,
        sort_by=["response", "balanced_accuracy"],
        ascending=[True, False],
    ))
    lines.append("")

    lines.append("At the 90% adequacy cutoff, global diagnostic thresholds based on total relative P uptake clearly outperformed thresholds based on total relative dry matter. Olsen, Resin, and Mehlich-3 had similar diagnostic performance for total relative P uptake, while Mehlich-1 was weaker.")

    add_section(lines, "2. By-soil threshold stability at 90% adequacy")

    by_soil_90 = by_soil[
        (by_soil["adequacy_cutoff_pct"] == 90.0)
        & (by_soil["response"].isin(PRIMARY_RESPONSES))
    ].copy()

    cols = [
        "response",
        "extractor",
        "n_soils",
        "median_threshold_mg_dm3",
        "min_threshold_mg_dm3",
        "max_threshold_mg_dm3",
        "cv_threshold",
        "median_balanced_accuracy",
        "n_balanced_accuracy_ge_0_70",
        "n_balanced_accuracy_ge_0_80",
    ]

    lines.append(fmt_table(
        by_soil_90,
        cols,
        sort_by=["response", "median_balanced_accuracy"],
        ascending=[True, False],
    ))
    lines.append("")

    lines.append("The by-soil results indicate that total relative P uptake generated high within-soil diagnostic performance. Olsen had the highest median balanced accuracy, while Mehlich-3 had a comparatively lower coefficient of variation for the threshold, indicating stronger threshold stability among soils.")

    add_section(lines, "3. Logistic/probabilistic model performance")

    log90 = logistic[
        (logistic["adequacy_cutoff_pct"] == 90.0)
        & (logistic["response"].isin(["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]))
    ].copy()

    cols = [
        "response",
        "extractor",
        "model_type",
        "n",
        "n_adequate",
        "n_inadequate",
        "auc",
        "brier_score",
        "pseudo_r2_mcfadden",
        "slope",
        "slope_p_value",
        "aic",
    ]

    lines.append(fmt_table(
        log90,
        cols,
        sort_by=["response", "model_type", "auc"],
        ascending=[True, True, False],
    ))
    lines.append("")

    lines.append("Logistic models confirmed that total relative P uptake was more predictable from extracted P than total relative dry matter. For total relative P uptake, soil-fixed logistic models had higher AUC values, especially for Resin and Mehlich-3. For dry matter, AUC values and slope significance were weak.")

    add_section(lines, "4. Observed-range check for probabilistic thresholds")

    range_90 = range_check[
        (range_check["adequacy_cutoff_pct"] == 90.0)
        & (range_check["response"].isin(["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]))
        & (range_check["model_type"] == "soil_fixed_logistic")
        & (range_check["probability_level"].isin([0.50, 0.75]))
    ].copy()

    range_counts = (
        range_90
        .groupby(["response", "probability_level", "extractor", "range_status"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )

    lines.append(fmt_table(
        range_counts,
        ["response", "probability_level", "extractor", "range_status", "n"],
        sort_by=["response", "probability_level", "extractor", "range_status"],
    ))
    lines.append("")

    lines.append("Most probabilistic thresholds, especially at 50% and 75% predicted adequacy, were above the observed extractor range. This means the probabilistic models are useful for comparing predictability, but their estimated probability thresholds should not be treated as direct agronomic critical values.")

    add_section(lines, "5. Bootstrap stability of global diagnostic thresholds")

    boot90 = bootstrap[
        bootstrap["response"].isin(PRIMARY_RESPONSES)
    ].copy()

    cols = [
        "response",
        "extractor",
        "observed_threshold_mg_dm3",
        "observed_balanced_accuracy",
        "median_boot_threshold_mg_dm3",
        "q025_threshold_mg_dm3",
        "q975_threshold_mg_dm3",
        "relative_ci_width",
        "median_boot_balanced_accuracy",
        "q025_balanced_accuracy",
        "q975_balanced_accuracy",
    ]

    lines.append(fmt_table(
        boot90,
        cols,
        sort_by=["response", "median_boot_balanced_accuracy"],
        ascending=[True, False],
    ))
    lines.append("")

    lines.append("Bootstrap analysis supported the ranking of total relative P uptake as the strongest response basis, but global threshold intervals were wide. Mehlich-3 showed comparatively better threshold stability for total relative P uptake than Olsen and Mehlich-1, even when its median balanced accuracy was not the highest.")

    add_section(lines, "6. Leave-one-soil-out validation")

    loso90 = loso[
        (loso["adequacy_cutoff_pct"] == 90.0)
        & (loso["response"].isin(PRIMARY_RESPONSES))
    ].copy()

    cols = [
        "response",
        "extractor",
        "n_validations",
        "n_estimable_thresholds",
        "median_threshold_mg_dm3",
        "median_training_balanced_accuracy",
        "median_validation_balanced_accuracy",
        "mean_validation_balanced_accuracy",
        "min_validation_balanced_accuracy",
        "n_validation_ba_ge_0_60",
        "n_validation_ba_ge_0_70",
        "n_validation_ba_ge_0_80",
    ]

    lines.append(fmt_table(
        loso90,
        cols,
        sort_by=["response", "median_validation_balanced_accuracy"],
        ascending=[True, False],
    ))
    lines.append("")

    lines.append("Leave-one-soil-out validation showed that global thresholds had limited transferability to held-out soils. This is the strongest caution against proposing a universal critical P threshold from this dataset. The result supports soil-specific calibration or explicit soil-effect control.")

    add_section(lines, "7. Response-specific outlier sensitivity")

    sens90 = sensitivity[
        (sensitivity["adequacy_cutoff_pct"] == 90.0)
        & (sensitivity["response"].isin(PRIMARY_RESPONSES))
    ].copy()

    cols = [
        "response",
        "extractor",
        "n_unique_response_outliers",
        "all_data_n",
        "without_outliers_n",
        "all_data_threshold_mg_dm3",
        "without_outliers_threshold_mg_dm3",
        "threshold_change_mg_dm3",
        "threshold_relative_change_pct",
        "all_data_balanced_accuracy",
        "without_outliers_balanced_accuracy",
        "balanced_accuracy_change",
    ]

    lines.append(fmt_table(
        sens90,
        cols,
        sort_by=["response", "without_outliers_balanced_accuracy"],
        ascending=[True, False],
    ))
    lines.append("")

    lines.append("Response-specific outlier sensitivity showed that the primary result for total relative P uptake was robust. Removing the response-specific outlier did not change the global thresholds for Olsen, Resin, Mehlich-3, or Mehlich-1. First-cut P uptake was more sensitive and should remain secondary.")

    add_section(lines, "Working conclusion")

    lines.append("The current evidence supports the following internal conclusion:")
    lines.append("")
    lines.append("1. Total relative P uptake is the most defensible biological response for threshold reassessment in this dataset.")
    lines.append("2. Dry matter, even as relative dry matter, is a weaker basis for critical P thresholds.")
    lines.append("3. Olsen can classify high relative P uptake well in diagnostic threshold analysis, but Mehlich-3 shows a better balance of stability, soil-effect control, and consistency with previous extractor-performance interpretations.")
    lines.append("4. Probabilistic thresholds should not be interpreted as direct critical values because many exceed the observed extractor range.")
    lines.append("5. Leave-one-soil-out validation indicates that threshold transferability among soils is limited.")
    lines.append("6. A future manuscript should avoid claiming a universal critical P value. The stronger framing is methodological: response-variable choice and soil specificity affect P extractor thresholds in alkaline soils.")
    lines.append("")
    lines.append("Possible manuscript framing:")
    lines.append("")
    lines.append("> Response-variable choice and soil specificity affect phosphorus extractor thresholds in alkaline soils: evidence from relative P uptake and diagnostic threshold reassessment.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Private synthesis written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
