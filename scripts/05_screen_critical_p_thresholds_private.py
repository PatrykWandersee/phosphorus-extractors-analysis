from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"
TABLES_DIR = BASE_DIR / "tables/private"

THRESHOLDS_PATH = TABLES_DIR / "critical_p_thresholds_screening.csv"
THRESHOLD_SUMMARY_PATH = TABLES_DIR / "critical_p_thresholds_summary.csv"


RESPONSE_REL_COLUMNS = [
    "dry_matter_cut1_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_total_rel_pct",
]

ADEQUACY_CUTOFF = 90.0


def balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    positives = y_true == 1
    negatives = y_true == 0

    sensitivity = np.nan
    specificity = np.nan

    if positives.sum() > 0:
        sensitivity = ((y_pred == 1) & positives).sum() / positives.sum()

    if negatives.sum() > 0:
        specificity = ((y_pred == 0) & negatives).sum() / negatives.sum()

    values = [v for v in [sensitivity, specificity] if not np.isnan(v)]

    if not values:
        return np.nan

    return float(np.mean(values))


def screen_threshold(subset: pd.DataFrame, response_col: str) -> dict:
    clean = subset[["P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) < 6:
        return {
            "n": len(clean),
            "n_adequate": np.nan,
            "n_low": np.nan,
            "threshold_mg_dm3": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        }

    clean["adequate"] = (clean[response_col] >= ADEQUACY_CUTOFF).astype(int)

    n_adequate = int(clean["adequate"].sum())
    n_low = int((clean["adequate"] == 0).sum())

    if n_adequate == 0 or n_low == 0:
        return {
            "n": len(clean),
            "n_adequate": n_adequate,
            "n_low": n_low,
            "threshold_mg_dm3": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        }

    candidate_thresholds = sorted(clean["P_extracted_mg_dm3"].unique())

    best = None

    for threshold in candidate_thresholds:
        predicted = (clean["P_extracted_mg_dm3"] >= threshold).astype(int)

        positives = clean["adequate"] == 1
        negatives = clean["adequate"] == 0

        sensitivity = ((predicted == 1) & positives).sum() / positives.sum()
        specificity = ((predicted == 0) & negatives).sum() / negatives.sum()
        bal_acc = balanced_accuracy(clean["adequate"], predicted)

        result = {
            "threshold_mg_dm3": threshold,
            "balanced_accuracy": bal_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

        if best is None:
            best = result
            continue

        if result["balanced_accuracy"] > best["balanced_accuracy"]:
            best = result
        elif result["balanced_accuracy"] == best["balanced_accuracy"]:
            # Prefer the lower threshold when diagnostic performance is tied.
            if result["threshold_mg_dm3"] < best["threshold_mg_dm3"]:
                best = result

    return {
        "n": len(clean),
        "n_adequate": n_adequate,
        "n_low": n_low,
        **best,
    }


def compute_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["soil_id", "soil_label", "extractor"]

    for (soil_id, soil_label, extractor), group in df.groupby(group_cols):
        for response_col in RESPONSE_REL_COLUMNS:
            result = screen_threshold(group, response_col)

            rows.append({
                "soil_id": soil_id,
                "soil_label": soil_label,
                "extractor": extractor,
                "response_rel": response_col,
                "adequacy_cutoff_pct": ADEQUACY_CUTOFF,
                **result,
            })

    return pd.DataFrame(rows)


def summarize_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    clean = thresholds.dropna(subset=["threshold_mg_dm3"]).copy()

    summary = (
        clean
        .groupby(["extractor", "response_rel"], as_index=False)
        .agg(
            n_soils=("soil_id", "count"),
            median_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            mean_threshold_mg_dm3=("threshold_mg_dm3", "mean"),
            min_threshold_mg_dm3=("threshold_mg_dm3", "min"),
            max_threshold_mg_dm3=("threshold_mg_dm3", "max"),
            median_balanced_accuracy=("balanced_accuracy", "median"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            n_balanced_accuracy_ge_0_70=("balanced_accuracy", lambda x: (x >= 0.70).sum()),
        )
    )

    return summary.sort_values(
        ["response_rel", "median_balanced_accuracy"],
        ascending=[True, False],
    )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Extractor long-format dataset not found: {EXTRACTOR_LONG_PATH}\n"
            "Run scripts/04_prepare_relative_response_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    thresholds = compute_thresholds(df)
    summary = summarize_thresholds(thresholds)

    thresholds.to_csv(THRESHOLDS_PATH, index=False)
    summary.to_csv(THRESHOLD_SUMMARY_PATH, index=False)

    print("Critical P threshold screening finished.")
    print(f"Thresholds by soil: {THRESHOLDS_PATH}")
    print(f"Summary: {THRESHOLD_SUMMARY_PATH}")

    for response in ["dry_matter_total_rel_pct", "P_uptake_total_rel_pct", "dry_matter_cut1_rel_pct", "P_uptake_cut1_rel_pct"]:
        print(f"\nThreshold summary for {response}:")
        subset = summary[summary["response_rel"] == response].copy()
        print(
            subset[[
                "extractor",
                "n_soils",
                "median_threshold_mg_dm3",
                "min_threshold_mg_dm3",
                "max_threshold_mg_dm3",
                "median_balanced_accuracy",
                "n_balanced_accuracy_ge_0_70",
            ]]
            .sort_values("median_balanced_accuracy", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
