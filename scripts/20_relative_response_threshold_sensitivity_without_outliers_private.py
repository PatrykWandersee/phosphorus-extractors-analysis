from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"
OUTLIERS_PATH = BASE_DIR / "tables/private/plant_response_rstudent_outliers.csv"

TABLES_DIR = BASE_DIR / "tables/private"
SENSITIVITY_RESULTS_PATH = TABLES_DIR / "relative_response_threshold_sensitivity_without_outliers.csv"
SENSITIVITY_SUMMARY_PATH = TABLES_DIR / "relative_response_threshold_sensitivity_summary_without_outliers.csv"


EXTRACTORS = {
    "Mehlich-1": "P_Mehlich1_mg_dm3",
    "Mehlich-3": "P_Mehlich3_mg_dm3",
    "Resin": "P_resin_mg_dm3",
    "Olsen": "P_Olsen_mg_dm3",
}

RESPONSES = {
    "P_uptake_total_rel_pct": "P_uptake_total_mg_pot",
    "dry_matter_total_rel_pct": "dry_matter_total_g",
    "P_uptake_cut1_rel_pct": "P_uptake_cut1_mg_pot",
    "dry_matter_cut1_rel_pct": "dry_matter_cut1_g",
}

ADEQUACY_CUTOFFS = [80.0, 90.0, 95.0]


def build_threshold_candidates(values: pd.Series) -> np.ndarray:
    unique_values = np.sort(values.dropna().unique())

    if len(unique_values) == 0:
        return np.array([])

    if len(unique_values) == 1:
        return np.array([unique_values[0]])

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    lower = unique_values[0] - 1e-9
    upper = unique_values[-1] + 1e-9

    return np.concatenate([[lower], midpoints, [upper]])


def diagnostic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if pd.notna(sensitivity) and pd.notna(specificity):
        balanced_accuracy = (sensitivity + specificity) / 2
        youden_j = sensitivity + specificity - 1
    else:
        balanced_accuracy = np.nan
        youden_j = np.nan

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else np.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
    }


def find_best_threshold(data: pd.DataFrame, response_col: str, cutoff: float) -> dict:
    clean = data[["P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) < 6:
        return {
            "n": len(clean),
            "n_adequate": np.nan,
            "n_inadequate": np.nan,
            "threshold_mg_dm3": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "accuracy": np.nan,
            "youden_j": np.nan,
            "tp": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }

    y_true = (clean[response_col].to_numpy(dtype=float) >= cutoff).astype(int)
    x = clean["P_extracted_mg_dm3"].to_numpy(dtype=float)

    n_adequate = int(np.sum(y_true == 1))
    n_inadequate = int(np.sum(y_true == 0))

    if n_adequate == 0 or n_inadequate == 0:
        return {
            "n": len(clean),
            "n_adequate": n_adequate,
            "n_inadequate": n_inadequate,
            "threshold_mg_dm3": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "accuracy": np.nan,
            "youden_j": np.nan,
            "tp": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }

    candidates = build_threshold_candidates(clean["P_extracted_mg_dm3"])

    rows = []
    for threshold in candidates:
        y_pred = (x >= threshold).astype(int)
        metrics = diagnostic_metrics(y_true, y_pred)
        rows.append({
            "threshold_mg_dm3": threshold,
            **metrics,
        })

    results = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "youden_j", "accuracy", "threshold_mg_dm3"],
        ascending=[False, False, False, True],
    )

    best = results.iloc[0].to_dict()

    return {
        "n": len(clean),
        "n_adequate": n_adequate,
        "n_inadequate": n_inadequate,
        **best,
    }


def get_outlier_sample_codes(outliers: pd.DataFrame, raw_response: str) -> set:
    subset = outliers[outliers["response"] == raw_response].copy()
    return set(subset["sample_code"].astype(int).unique())


def make_relative_response_dataset(
    df: pd.DataFrame,
    response_name: str,
    raw_response: str,
    outlier_codes: set,
    scenario: str,
) -> pd.DataFrame:
    work = df.copy()

    if scenario == "without_response_outliers":
        work = work[~work["sample_code"].astype(int).isin(outlier_codes)].copy()

    max_by_soil = (
        work
        .groupby("soil_id")[raw_response]
        .transform("max")
    )

    work[response_name] = np.where(
        max_by_soil > 0,
        work[raw_response] / max_by_soil * 100,
        np.nan,
    )

    long_rows = []

    for extractor_name, extractor_col in EXTRACTORS.items():
        temp = work[
            [
                "sample_code",
                "soil_id",
                "soil_label",
                extractor_col,
                response_name,
            ]
        ].copy()

        temp = temp.rename(columns={extractor_col: "P_extracted_mg_dm3"})
        temp["extractor"] = extractor_name
        temp["response"] = response_name
        temp["raw_response"] = raw_response
        temp["scenario"] = scenario
        temp["n_response_outliers_removed"] = len(outlier_codes)

        long_rows.append(temp)

    return pd.concat(long_rows, ignore_index=True)


def run_sensitivity(df: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for response_name, raw_response in RESPONSES.items():
        outlier_codes = get_outlier_sample_codes(outliers, raw_response)

        for scenario in ["all_data", "without_response_outliers"]:
            long_df = make_relative_response_dataset(
                df=df,
                response_name=response_name,
                raw_response=raw_response,
                outlier_codes=outlier_codes,
                scenario=scenario,
            )

            for cutoff in ADEQUACY_CUTOFFS:
                for extractor, group in long_df.groupby("extractor"):
                    result = find_best_threshold(group, response_name, cutoff)

                    rows.append({
                        "adequacy_cutoff_pct": cutoff,
                        "response": response_name,
                        "raw_response": raw_response,
                        "extractor": extractor,
                        "scenario": scenario,
                        "n_unique_response_outliers": len(outlier_codes),
                        **result,
                    })

    return pd.DataFrame(rows)


def summarize_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    all_data = results[results["scenario"] == "all_data"].copy()
    no_outliers = results[results["scenario"] == "without_response_outliers"].copy()

    key_cols = ["adequacy_cutoff_pct", "response", "raw_response", "extractor"]

    merged = no_outliers.merge(
        all_data[
            key_cols
            + [
                "threshold_mg_dm3",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "n",
                "n_adequate",
                "n_inadequate",
            ]
        ].rename(columns={
            "threshold_mg_dm3": "all_data_threshold_mg_dm3",
            "balanced_accuracy": "all_data_balanced_accuracy",
            "sensitivity": "all_data_sensitivity",
            "specificity": "all_data_specificity",
            "n": "all_data_n",
            "n_adequate": "all_data_n_adequate",
            "n_inadequate": "all_data_n_inadequate",
        }),
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    merged = merged.rename(columns={
        "threshold_mg_dm3": "without_outliers_threshold_mg_dm3",
        "balanced_accuracy": "without_outliers_balanced_accuracy",
        "sensitivity": "without_outliers_sensitivity",
        "specificity": "without_outliers_specificity",
        "n": "without_outliers_n",
        "n_adequate": "without_outliers_n_adequate",
        "n_inadequate": "without_outliers_n_inadequate",
    })

    merged["threshold_change_mg_dm3"] = (
        merged["without_outliers_threshold_mg_dm3"]
        - merged["all_data_threshold_mg_dm3"]
    )

    merged["threshold_relative_change_pct"] = np.where(
        merged["all_data_threshold_mg_dm3"].abs() > 0,
        merged["threshold_change_mg_dm3"] / merged["all_data_threshold_mg_dm3"] * 100,
        np.nan,
    )

    merged["balanced_accuracy_change"] = (
        merged["without_outliers_balanced_accuracy"]
        - merged["all_data_balanced_accuracy"]
    )

    return merged.sort_values(
        ["adequacy_cutoff_pct", "response", "without_outliers_balanced_accuracy"],
        ascending=[True, True, False],
    )


def print_primary_summary(summary: pd.DataFrame) -> None:
    cutoff = 90.0
    selected = summary[summary["adequacy_cutoff_pct"] == cutoff].copy()

    for response in [
        "P_uptake_total_rel_pct",
        "dry_matter_total_rel_pct",
        "P_uptake_cut1_rel_pct",
        "dry_matter_cut1_rel_pct",
    ]:
        print(f"\n=== {response} | 90% adequacy | all data vs without response-specific outliers ===")

        subset = selected[selected["response"] == response].copy()

        print(
            subset[
                [
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
            ]
            .sort_values(
                ["without_outliers_balanced_accuracy", "extractor"],
                ascending=[False, True],
            )
            .to_string(index=False)
        )


def main() -> None:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}"
        )

    if not OUTLIERS_PATH.exists():
        raise FileNotFoundError(
            f"Outlier table not found: {OUTLIERS_PATH}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DATASET_PATH)
    outliers = pd.read_csv(OUTLIERS_PATH)

    results = run_sensitivity(df, outliers)
    summary = summarize_sensitivity(results)

    results.to_csv(SENSITIVITY_RESULTS_PATH, index=False)
    summary.to_csv(SENSITIVITY_SUMMARY_PATH, index=False)

    print("Relative response threshold sensitivity without outliers finished.")
    print(f"Sensitivity results: {SENSITIVITY_RESULTS_PATH}")
    print(f"Sensitivity summary: {SENSITIVITY_SUMMARY_PATH}")

    print_primary_summary(summary)


if __name__ == "__main__":
    main()
