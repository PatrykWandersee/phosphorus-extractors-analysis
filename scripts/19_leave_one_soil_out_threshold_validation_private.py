from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
LOSO_RESULTS_PATH = TABLES_DIR / "relative_response_threshold_leave_one_soil_out_results.csv"
LOSO_SUMMARY_PATH = TABLES_DIR / "relative_response_threshold_leave_one_soil_out_summary.csv"


RESPONSE_COLUMNS = [
    "P_uptake_total_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_cut1_rel_pct",
]

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
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
        "accuracy": accuracy,
    }


def find_best_threshold(training: pd.DataFrame, response_col: str, cutoff: float) -> dict:
    clean = training[["P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) < 10:
        return {
            "threshold_mg_dm3": np.nan,
            "training_balanced_accuracy": np.nan,
            "training_sensitivity": np.nan,
            "training_specificity": np.nan,
            "training_n": len(clean),
            "training_n_adequate": np.nan,
            "training_n_inadequate": np.nan,
        }

    y_true = (clean[response_col].to_numpy(dtype=float) >= cutoff).astype(int)
    x = clean["P_extracted_mg_dm3"].to_numpy(dtype=float)

    n_adequate = int(np.sum(y_true == 1))
    n_inadequate = int(np.sum(y_true == 0))

    if n_adequate == 0 or n_inadequate == 0:
        return {
            "threshold_mg_dm3": np.nan,
            "training_balanced_accuracy": np.nan,
            "training_sensitivity": np.nan,
            "training_specificity": np.nan,
            "training_n": len(clean),
            "training_n_adequate": n_adequate,
            "training_n_inadequate": n_inadequate,
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

    best = results.iloc[0]

    return {
        "threshold_mg_dm3": best["threshold_mg_dm3"],
        "training_balanced_accuracy": best["balanced_accuracy"],
        "training_sensitivity": best["sensitivity"],
        "training_specificity": best["specificity"],
        "training_n": len(clean),
        "training_n_adequate": n_adequate,
        "training_n_inadequate": n_inadequate,
    }


def evaluate_threshold(validation: pd.DataFrame, response_col: str, cutoff: float, threshold: float) -> dict:
    clean = validation[["P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) == 0 or pd.isna(threshold):
        return {
            "validation_n": len(clean),
            "validation_n_adequate": np.nan,
            "validation_n_inadequate": np.nan,
            "validation_balanced_accuracy": np.nan,
            "validation_sensitivity": np.nan,
            "validation_specificity": np.nan,
            "validation_accuracy": np.nan,
            "tp": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }

    y_true = (clean[response_col].to_numpy(dtype=float) >= cutoff).astype(int)
    x = clean["P_extracted_mg_dm3"].to_numpy(dtype=float)
    y_pred = (x >= threshold).astype(int)

    metrics = diagnostic_metrics(y_true, y_pred)

    return {
        "validation_n": len(clean),
        "validation_n_adequate": int(np.sum(y_true == 1)),
        "validation_n_inadequate": int(np.sum(y_true == 0)),
        "validation_balanced_accuracy": metrics["balanced_accuracy"],
        "validation_sensitivity": metrics["sensitivity"],
        "validation_specificity": metrics["specificity"],
        "validation_accuracy": metrics["accuracy"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }


def leave_one_soil_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    soils = (
        df[["soil_id", "soil_label"]]
        .drop_duplicates()
        .sort_values("soil_id")
        .to_dict("records")
    )

    for cutoff in ADEQUACY_CUTOFFS:
        for response in RESPONSE_COLUMNS:
            for extractor, extractor_df in df.groupby("extractor"):
                for soil in soils:
                    held_soil_id = soil["soil_id"]
                    held_soil_label = soil["soil_label"]

                    training = extractor_df[extractor_df["soil_id"] != held_soil_id].copy()
                    validation = extractor_df[extractor_df["soil_id"] == held_soil_id].copy()

                    best = find_best_threshold(training, response, cutoff)
                    evaluation = evaluate_threshold(
                        validation,
                        response,
                        cutoff,
                        best["threshold_mg_dm3"],
                    )

                    rows.append({
                        "adequacy_cutoff_pct": cutoff,
                        "response": response,
                        "extractor": extractor,
                        "held_out_soil_id": held_soil_id,
                        "held_out_soil_label": held_soil_label,
                        **best,
                        **evaluation,
                    })

    return pd.DataFrame(rows)


def summarize_loso(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results
        .groupby(["adequacy_cutoff_pct", "response", "extractor"], as_index=False)
        .agg(
            n_validations=("held_out_soil_label", "count"),
            n_estimable_thresholds=("threshold_mg_dm3", lambda x: int(x.notna().sum())),
            median_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            min_threshold_mg_dm3=("threshold_mg_dm3", "min"),
            max_threshold_mg_dm3=("threshold_mg_dm3", "max"),
            median_training_balanced_accuracy=("training_balanced_accuracy", "median"),
            median_validation_balanced_accuracy=("validation_balanced_accuracy", "median"),
            mean_validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
            min_validation_balanced_accuracy=("validation_balanced_accuracy", "min"),
            max_validation_balanced_accuracy=("validation_balanced_accuracy", "max"),
            n_validation_ba_ge_0_60=("validation_balanced_accuracy", lambda x: int((x >= 0.60).sum())),
            n_validation_ba_ge_0_70=("validation_balanced_accuracy", lambda x: int((x >= 0.70).sum())),
            n_validation_ba_ge_0_80=("validation_balanced_accuracy", lambda x: int((x >= 0.80).sum())),
        )
    )

    return summary.sort_values(
        [
            "adequacy_cutoff_pct",
            "response",
            "median_validation_balanced_accuracy",
            "n_validation_ba_ge_0_70",
        ],
        ascending=[True, True, False, False],
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
        print(f"\n=== {response} | leave-one-soil-out validation | 90% adequacy ===")

        subset = selected[selected["response"] == response].copy()

        print(
            subset[
                [
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
            ]
            .sort_values(["median_validation_balanced_accuracy", "n_validation_ba_ge_0_70"], ascending=[False, False])
            .to_string(index=False)
        )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Long-format dataset not found: {EXTRACTOR_LONG_PATH}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    results = leave_one_soil_out(df)
    summary = summarize_loso(results)

    results.to_csv(LOSO_RESULTS_PATH, index=False)
    summary.to_csv(LOSO_SUMMARY_PATH, index=False)

    print("Leave-one-soil-out threshold validation finished.")
    print(f"LOSO results: {LOSO_RESULTS_PATH}")
    print(f"LOSO summary: {LOSO_SUMMARY_PATH}")

    print_primary_summary(summary)


if __name__ == "__main__":
    main()
