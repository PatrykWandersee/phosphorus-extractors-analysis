from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
BOOTSTRAP_RESULTS_PATH = TABLES_DIR / "relative_response_threshold_bootstrap_results.csv"
BOOTSTRAP_SUMMARY_PATH = TABLES_DIR / "relative_response_threshold_bootstrap_summary.csv"


RANDOM_SEED = 42
N_BOOTSTRAPS = 1000

RESPONSE_COLUMNS = [
    "P_uptake_total_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_cut1_rel_pct",
]

EXTRACTORS = [
    "Mehlich-1",
    "Mehlich-3",
    "Resin",
    "Olsen",
]

ADEQUACY_CUTOFF = 90.0


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

    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan

    return {
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def find_best_threshold(data: pd.DataFrame, response_col: str) -> dict:
    clean = data[["P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) < 6:
        return empty_result(len(clean))

    y_true = (clean[response_col].to_numpy(dtype=float) >= ADEQUACY_CUTOFF).astype(int)
    x = clean["P_extracted_mg_dm3"].to_numpy(dtype=float)

    n_adequate = int(np.sum(y_true == 1))
    n_inadequate = int(np.sum(y_true == 0))

    if n_adequate == 0 or n_inadequate == 0:
        out = empty_result(len(clean))
        out["n_adequate"] = n_adequate
        out["n_inadequate"] = n_inadequate
        return out

    candidates = build_threshold_candidates(clean["P_extracted_mg_dm3"])

    rows = []
    for threshold in candidates:
        y_pred = (x >= threshold).astype(int)
        rows.append({
            "threshold_mg_dm3": threshold,
            **diagnostic_metrics(y_true, y_pred),
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


def empty_result(n: int) -> dict:
    return {
        "n": n,
        "n_adequate": np.nan,
        "n_inadequate": np.nan,
        "threshold_mg_dm3": np.nan,
        "balanced_accuracy": np.nan,
        "youden_j": np.nan,
        "accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
    }


def bootstrap_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for response in RESPONSE_COLUMNS:
        for extractor in EXTRACTORS:
            subset = df[df["extractor"] == extractor].dropna(
                subset=["P_extracted_mg_dm3", response]
            ).copy()

            if subset.empty:
                continue

            observed = find_best_threshold(subset, response)
            rows.append({
                "response": response,
                "extractor": extractor,
                "bootstrap_id": "observed",
                **observed,
            })

            for i in range(N_BOOTSTRAPS):
                sampled_indices = rng.choice(
                    subset.index.to_numpy(),
                    size=len(subset),
                    replace=True,
                )
                sampled = subset.loc[sampled_indices].copy()

                result = find_best_threshold(sampled, response)
                rows.append({
                    "response": response,
                    "extractor": extractor,
                    "bootstrap_id": i,
                    **result,
                })

    return pd.DataFrame(rows)


def summarize_bootstrap(results: pd.DataFrame) -> pd.DataFrame:
    boot = results[results["bootstrap_id"] != "observed"].dropna(
        subset=["threshold_mg_dm3"]
    ).copy()

    observed = results[results["bootstrap_id"] == "observed"].copy()

    summary = (
        boot
        .groupby(["response", "extractor"], as_index=False)
        .agg(
            n_bootstrap_valid=("threshold_mg_dm3", "count"),
            median_boot_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            q025_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.025)),
            q975_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.975)),
            q25_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.25)),
            q75_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.75)),
            median_boot_balanced_accuracy=("balanced_accuracy", "median"),
            q025_balanced_accuracy=("balanced_accuracy", lambda x: x.quantile(0.025)),
            q975_balanced_accuracy=("balanced_accuracy", lambda x: x.quantile(0.975)),
        )
    )

    summary["iqr_threshold_mg_dm3"] = (
        summary["q75_threshold_mg_dm3"] - summary["q25_threshold_mg_dm3"]
    )

    summary = summary.merge(
        observed[
            [
                "response",
                "extractor",
                "threshold_mg_dm3",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "n",
                "n_adequate",
                "n_inadequate",
            ]
        ].rename(columns={
            "threshold_mg_dm3": "observed_threshold_mg_dm3",
            "balanced_accuracy": "observed_balanced_accuracy",
            "sensitivity": "observed_sensitivity",
            "specificity": "observed_specificity",
            "n": "observed_n",
            "n_adequate": "observed_n_adequate",
            "n_inadequate": "observed_n_inadequate",
        }),
        on=["response", "extractor"],
        how="left",
    )

    summary["relative_ci_width"] = (
        (summary["q975_threshold_mg_dm3"] - summary["q025_threshold_mg_dm3"])
        / summary["median_boot_threshold_mg_dm3"]
    )

    return summary.sort_values(
        ["response", "median_boot_balanced_accuracy", "relative_ci_width"],
        ascending=[True, False, True],
    )


def print_primary_summary(summary: pd.DataFrame) -> None:
    for response in [
        "P_uptake_total_rel_pct",
        "dry_matter_total_rel_pct",
        "P_uptake_cut1_rel_pct",
        "dry_matter_cut1_rel_pct",
    ]:
        print(f"\n=== {response} | 90% adequacy cutoff | bootstrap global thresholds ===")
        subset = summary[summary["response"] == response].copy()

        print(
            subset[
                [
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
                    "n_bootstrap_valid",
                ]
            ]
            .sort_values(["median_boot_balanced_accuracy", "relative_ci_width"], ascending=[False, True])
            .to_string(index=False)
        )


def main() -> None:
    if not EXTRACTOR_LONG_PATH.exists():
        raise FileNotFoundError(
            f"Long-format dataset not found: {EXTRACTOR_LONG_PATH}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EXTRACTOR_LONG_PATH)

    results = bootstrap_thresholds(df)
    summary = summarize_bootstrap(results)

    results.to_csv(BOOTSTRAP_RESULTS_PATH, index=False)
    summary.to_csv(BOOTSTRAP_SUMMARY_PATH, index=False)

    print("Bootstrap relative response threshold analysis finished.")
    print(f"Bootstrap results: {BOOTSTRAP_RESULTS_PATH}")
    print(f"Bootstrap summary: {BOOTSTRAP_SUMMARY_PATH}")

    print_primary_summary(summary)


if __name__ == "__main__":
    main()
