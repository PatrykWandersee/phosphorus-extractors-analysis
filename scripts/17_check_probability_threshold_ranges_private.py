from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"
PROB_THRESHOLDS_PATH = BASE_DIR / "tables/private/relative_response_probability_thresholds.csv"

TABLES_DIR = BASE_DIR / "tables/private"
RANGE_CHECK_PATH = TABLES_DIR / "relative_response_probability_threshold_range_check.csv"
RANGE_SUMMARY_PATH = TABLES_DIR / "relative_response_probability_threshold_range_summary.csv"


PRIMARY_RESPONSES = [
    "P_uptake_total_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_cut1_rel_pct",
]


def build_observed_ranges(df: pd.DataFrame) -> pd.DataFrame:
    by_soil = (
        df
        .groupby(["extractor", "soil_label"], as_index=False)
        .agg(
            observed_min_mg_dm3=("P_extracted_mg_dm3", "min"),
            observed_max_mg_dm3=("P_extracted_mg_dm3", "max"),
            observed_median_mg_dm3=("P_extracted_mg_dm3", "median"),
            n_observed=("P_extracted_mg_dm3", "count"),
        )
    )

    pooled = (
        df
        .groupby(["extractor"], as_index=False)
        .agg(
            observed_min_mg_dm3=("P_extracted_mg_dm3", "min"),
            observed_max_mg_dm3=("P_extracted_mg_dm3", "max"),
            observed_median_mg_dm3=("P_extracted_mg_dm3", "median"),
            n_observed=("P_extracted_mg_dm3", "count"),
        )
    )
    pooled["soil_label"] = "pooled"

    return pd.concat([by_soil, pooled], ignore_index=True)


def classify_range(row: pd.Series) -> str:
    threshold = row["threshold_mg_dm3"]

    if pd.isna(threshold):
        return "not_estimable"

    if threshold < 0:
        return "negative"

    if threshold < row["observed_min_mg_dm3"]:
        return "below_observed_range"

    if threshold > row["observed_max_mg_dm3"]:
        return "above_observed_range"

    return "inside_observed_range"


def summarize_range_check(check: pd.DataFrame) -> pd.DataFrame:
    summary = (
        check
        .groupby(
            [
                "adequacy_cutoff_pct",
                "response",
                "extractor",
                "model_type",
                "probability_level",
                "range_status",
            ],
            as_index=False,
        )
        .agg(
            n_thresholds=("threshold_mg_dm3", "count"),
            median_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            min_threshold_mg_dm3=("threshold_mg_dm3", "min"),
            max_threshold_mg_dm3=("threshold_mg_dm3", "max"),
            median_observed_min_mg_dm3=("observed_min_mg_dm3", "median"),
            median_observed_max_mg_dm3=("observed_max_mg_dm3", "median"),
            median_auc=("auc", "median"),
            median_brier_score=("brier_score", "median"),
            median_slope_p_value=("slope_p_value", "median"),
        )
    )

    return summary.sort_values(
        [
            "adequacy_cutoff_pct",
            "response",
            "model_type",
            "probability_level",
            "extractor",
            "range_status",
        ]
    )


def print_primary_results(check: pd.DataFrame) -> None:
    primary = check[
        (check["adequacy_cutoff_pct"] == 90.0)
        & (check["probability_level"].isin([0.50, 0.75]))
        & (check["model_type"] == "soil_fixed_logistic")
        & (check["response"].isin(["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]))
    ].copy()

    print("\n=== Range status for soil-fixed logistic thresholds, 90% relative response ===")

    for response in ["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]:
        print(f"\nResponse: {response}")

        subset = primary[primary["response"] == response].copy()

        counts = (
            subset
            .groupby(["probability_level", "extractor", "range_status"])
            .size()
            .reset_index(name="n")
            .sort_values(["probability_level", "extractor", "range_status"])
        )

        print(counts.to_string(index=False))

    print("\n=== Detailed 50% probability thresholds for P_uptake_total_rel_pct ===")

    detailed = primary[
        (primary["response"] == "P_uptake_total_rel_pct")
        & (primary["probability_level"] == 0.50)
    ].copy()

    print(
        detailed[
            [
                "extractor",
                "soil_label",
                "threshold_mg_dm3",
                "observed_min_mg_dm3",
                "observed_max_mg_dm3",
                "range_status",
                "auc",
                "slope_p_value",
            ]
        ]
        .sort_values(["extractor", "soil_label"])
        .to_string(index=False)
    )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Long-format dataset not found: {EXTRACTOR_LONG_PATH}"
        )

    if not PROB_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Probability thresholds not found: {PROB_THRESHOLDS_PATH}\n"
            "Run scripts/16_probabilistic_relative_response_thresholds_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)
    thresholds = pd.read_csv(PROB_THRESHOLDS_PATH)

    observed_ranges = build_observed_ranges(df)

    check = thresholds.merge(
        observed_ranges,
        on=["extractor", "soil_label"],
        how="left",
        validate="many_to_one",
    )

    check["range_status"] = check.apply(classify_range, axis=1)

    summary = summarize_range_check(check)

    check.to_csv(RANGE_CHECK_PATH, index=False)
    summary.to_csv(RANGE_SUMMARY_PATH, index=False)

    print("Probability threshold range check finished.")
    print(f"Range check: {RANGE_CHECK_PATH}")
    print(f"Range summary: {RANGE_SUMMARY_PATH}")

    print_primary_results(check)


if __name__ == "__main__":
    main()
