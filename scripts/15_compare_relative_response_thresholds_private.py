from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
GLOBAL_THRESHOLDS_PATH = TABLES_DIR / "relative_response_global_thresholds.csv"
SOIL_THRESHOLDS_PATH = TABLES_DIR / "relative_response_thresholds_by_soil.csv"
THRESHOLD_SUMMARY_PATH = TABLES_DIR / "relative_response_threshold_summary_by_soil.csv"


RESPONSE_COLUMNS = [
    "dry_matter_total_rel_pct",
    "P_uptake_total_rel_pct",
    "dry_matter_cut1_rel_pct",
    "P_uptake_cut1_rel_pct",
]

ADEQUACY_CUTOFFS = [80.0, 90.0, 95.0]


def build_threshold_candidates(values: pd.Series) -> np.ndarray:
    unique_values = np.sort(values.dropna().unique())

    if len(unique_values) == 0:
        return np.array([])

    if len(unique_values) == 1:
        v = unique_values[0]
        return np.array([v])

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2

    # Include boundary candidates to allow all/none classifications.
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
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else np.nan

    if pd.notna(sensitivity) and pd.notna(specificity):
        balanced_accuracy = (sensitivity + specificity) / 2
        youden_j = sensitivity + specificity - 1
    else:
        balanced_accuracy = np.nan
        youden_j = np.nan

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

    results = pd.DataFrame(rows)

    # Maximize balanced accuracy; if tied, prefer the lower threshold.
    results = results.sort_values(
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


def global_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cutoff in ADEQUACY_CUTOFFS:
        for response_col in RESPONSE_COLUMNS:
            for extractor, group in df.groupby("extractor"):
                result = find_best_threshold(group, response_col, cutoff)
                rows.append({
                    "scale": "global",
                    "adequacy_cutoff_pct": cutoff,
                    "response": response_col,
                    "extractor": extractor,
                    **result,
                })

    return pd.DataFrame(rows)


def by_soil_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cutoff in ADEQUACY_CUTOFFS:
        for response_col in RESPONSE_COLUMNS:
            for (soil_id, soil_label, extractor), group in df.groupby(["soil_id", "soil_label", "extractor"]):
                result = find_best_threshold(group, response_col, cutoff)
                rows.append({
                    "scale": "by_soil",
                    "adequacy_cutoff_pct": cutoff,
                    "response": response_col,
                    "soil_id": soil_id,
                    "soil_label": soil_label,
                    "extractor": extractor,
                    **result,
                })

    return pd.DataFrame(rows)


def summarize_by_soil_thresholds(soil_thresholds: pd.DataFrame) -> pd.DataFrame:
    clean = soil_thresholds.dropna(subset=["threshold_mg_dm3"]).copy()

    summary = (
        clean
        .groupby(["adequacy_cutoff_pct", "response", "extractor"], as_index=False)
        .agg(
            n_soils=("soil_label", "count"),
            median_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            mean_threshold_mg_dm3=("threshold_mg_dm3", "mean"),
            min_threshold_mg_dm3=("threshold_mg_dm3", "min"),
            max_threshold_mg_dm3=("threshold_mg_dm3", "max"),
            q1_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.25)),
            q3_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.75)),
            median_balanced_accuracy=("balanced_accuracy", "median"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            n_balanced_accuracy_ge_0_70=("balanced_accuracy", lambda x: int((x >= 0.70).sum())),
            n_balanced_accuracy_ge_0_80=("balanced_accuracy", lambda x: int((x >= 0.80).sum())),
        )
    )

    summary["iqr_threshold_mg_dm3"] = (
        summary["q3_threshold_mg_dm3"] - summary["q1_threshold_mg_dm3"]
    )

    summary["cv_threshold"] = (
        clean
        .groupby(["adequacy_cutoff_pct", "response", "extractor"])["threshold_mg_dm3"]
        .agg(lambda x: np.std(x, ddof=1) / np.mean(x) if np.mean(x) != 0 and len(x) > 1 else np.nan)
        .to_numpy()
    )

    summary = summary.sort_values(
        ["adequacy_cutoff_pct", "response", "median_balanced_accuracy", "cv_threshold"],
        ascending=[True, True, False, True],
    )

    return summary


def print_primary_results(global_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    cutoff = 90.0

    print("\n=== Global thresholds at 90% relative response ===")
    global_90 = global_df[global_df["adequacy_cutoff_pct"] == cutoff].copy()

    for response in ["P_uptake_total_rel_pct", "dry_matter_total_rel_pct", "P_uptake_cut1_rel_pct", "dry_matter_cut1_rel_pct"]:
        print(f"\nResponse: {response}")
        subset = global_90[global_90["response"] == response].copy()
        print(
            subset[
                [
                    "extractor",
                    "n",
                    "n_adequate",
                    "n_inadequate",
                    "threshold_mg_dm3",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "accuracy",
                ]
            ]
            .sort_values(["balanced_accuracy", "threshold_mg_dm3"], ascending=[False, True])
            .to_string(index=False)
        )

    print("\n=== By-soil threshold stability at 90% relative response ===")
    summary_90 = summary[summary["adequacy_cutoff_pct"] == cutoff].copy()

    for response in ["P_uptake_total_rel_pct", "dry_matter_total_rel_pct", "P_uptake_cut1_rel_pct", "dry_matter_cut1_rel_pct"]:
        print(f"\nResponse: {response}")
        subset = summary_90[summary_90["response"] == response].copy()
        print(
            subset[
                [
                    "extractor",
                    "n_soils",
                    "median_threshold_mg_dm3",
                    "min_threshold_mg_dm3",
                    "max_threshold_mg_dm3",
                    "iqr_threshold_mg_dm3",
                    "cv_threshold",
                    "median_balanced_accuracy",
                    "n_balanced_accuracy_ge_0_70",
                    "n_balanced_accuracy_ge_0_80",
                ]
            ]
            .sort_values(["median_balanced_accuracy", "cv_threshold"], ascending=[False, True])
            .to_string(index=False)
        )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Long-format dataset not found: {EXTRACTOR_LONG_PATH}\n"
            "Run scripts/04_prepare_relative_response_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    global_df = global_thresholds(df)
    soil_df = by_soil_thresholds(df)
    summary_df = summarize_by_soil_thresholds(soil_df)

    global_df.to_csv(GLOBAL_THRESHOLDS_PATH, index=False)
    soil_df.to_csv(SOIL_THRESHOLDS_PATH, index=False)
    summary_df.to_csv(THRESHOLD_SUMMARY_PATH, index=False)

    print("Relative response threshold comparison finished.")
    print(f"Global thresholds: {GLOBAL_THRESHOLDS_PATH}")
    print(f"By-soil thresholds: {SOIL_THRESHOLDS_PATH}")
    print(f"Threshold summary: {THRESHOLD_SUMMARY_PATH}")

    print_primary_results(global_df, summary_df)


if __name__ == "__main__":
    main()
