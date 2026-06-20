from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.special import logit
from scipy.stats import rankdata


BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTOR_LONG_PATH = BASE_DIR / "data/private/processed/extractor_long_format.csv"

TABLES_DIR = BASE_DIR / "tables/private"
MODEL_RESULTS_PATH = TABLES_DIR / "relative_response_logistic_model_results.csv"
PROB_THRESHOLDS_PATH = TABLES_DIR / "relative_response_probability_thresholds.csv"
PROB_THRESHOLDS_SUMMARY_PATH = TABLES_DIR / "relative_response_probability_thresholds_summary.csv"


RESPONSE_COLUMNS = [
    "P_uptake_total_rel_pct",
    "dry_matter_total_rel_pct",
    "P_uptake_cut1_rel_pct",
    "dry_matter_cut1_rel_pct",
]

ADEQUACY_CUTOFFS = [80.0, 90.0, 95.0]
PROBABILITY_LEVELS = [0.50, 0.75]


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = rankdata(y_score)
    pos_rank_sum = np.sum(ranks[y_true == 1])

    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def fit_glm(data: pd.DataFrame, formula: str):
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit()

    return model


def extract_soil_intercept(model, soil_label: str) -> float:
    intercept = model.params.get("Intercept", np.nan)

    if pd.isna(intercept):
        return np.nan

    term = f"C(soil_label)[T.{soil_label}]"
    return intercept + model.params.get(term, 0.0)


def probability_threshold(intercept: float, slope: float, probability: float) -> float:
    if pd.isna(intercept) or pd.isna(slope) or slope <= 0:
        return np.nan

    return float((logit(probability) - intercept) / slope)


def analyze_model(data: pd.DataFrame, response_col: str, cutoff: float, extractor: str, model_type: str):
    clean = data[["soil_label", "P_extracted_mg_dm3", response_col]].dropna().copy()

    if len(clean) < 12:
        return None, []

    clean["adequate"] = (clean[response_col] >= cutoff).astype(int)

    n_adequate = int(clean["adequate"].sum())
    n_inadequate = int((clean["adequate"] == 0).sum())

    if n_adequate == 0 or n_inadequate == 0:
        return None, []

    if model_type == "pooled_logistic":
        formula = "adequate ~ P_extracted_mg_dm3"
    elif model_type == "soil_fixed_logistic":
        formula = "adequate ~ P_extracted_mg_dm3 + C(soil_label)"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    try:
        model = fit_glm(clean, formula)
    except Exception as error:
        return {
            "adequacy_cutoff_pct": cutoff,
            "response": response_col,
            "extractor": extractor,
            "model_type": model_type,
            "n": len(clean),
            "n_adequate": n_adequate,
            "n_inadequate": n_inadequate,
            "error": str(error),
        }, []

    y_true = clean["adequate"].to_numpy()
    y_prob = model.predict(clean).to_numpy()

    slope = model.params.get("P_extracted_mg_dm3", np.nan)
    slope_p = model.pvalues.get("P_extracted_mg_dm3", np.nan)

    null_llf = model.llnull if hasattr(model, "llnull") else np.nan
    pseudo_r2_mcfadden = 1 - model.llf / null_llf if pd.notna(null_llf) and null_llf != 0 else np.nan

    model_row = {
        "adequacy_cutoff_pct": cutoff,
        "response": response_col,
        "extractor": extractor,
        "model_type": model_type,
        "n": len(clean),
        "n_adequate": n_adequate,
        "n_inadequate": n_inadequate,
        "aic": model.aic,
        "bic": getattr(model, "bic_llf", np.nan),
        "log_likelihood": model.llf,
        "pseudo_r2_mcfadden": pseudo_r2_mcfadden,
        "auc": auc_score(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "slope": slope,
        "slope_p_value": slope_p,
        "converged": getattr(model, "converged", np.nan),
        "error": "",
    }

    threshold_rows = []

    if model_type == "pooled_logistic":
        intercept = model.params.get("Intercept", np.nan)

        for probability in PROBABILITY_LEVELS:
            threshold_rows.append({
                "adequacy_cutoff_pct": cutoff,
                "response": response_col,
                "extractor": extractor,
                "model_type": model_type,
                "soil_label": "pooled",
                "probability_level": probability,
                "threshold_mg_dm3": probability_threshold(intercept, slope, probability),
                "slope": slope,
                "slope_p_value": slope_p,
                "n": len(clean),
                "n_adequate": n_adequate,
                "n_inadequate": n_inadequate,
                "auc": model_row["auc"],
                "brier_score": model_row["brier_score"],
            })

    if model_type == "soil_fixed_logistic":
        for soil_label in sorted(clean["soil_label"].unique()):
            soil_intercept = extract_soil_intercept(model, soil_label)

            for probability in PROBABILITY_LEVELS:
                threshold_rows.append({
                    "adequacy_cutoff_pct": cutoff,
                    "response": response_col,
                    "extractor": extractor,
                    "model_type": model_type,
                    "soil_label": soil_label,
                    "probability_level": probability,
                    "threshold_mg_dm3": probability_threshold(soil_intercept, slope, probability),
                    "slope": slope,
                    "slope_p_value": slope_p,
                    "n": len(clean),
                    "n_adequate": n_adequate,
                    "n_inadequate": n_inadequate,
                    "auc": model_row["auc"],
                    "brier_score": model_row["brier_score"],
                })

    return model_row, threshold_rows


def summarize_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    clean = thresholds.dropna(subset=["threshold_mg_dm3"]).copy()

    # Remove negative thresholds from summary because they are not useful critical values.
    clean = clean[clean["threshold_mg_dm3"] >= 0].copy()

    summary = (
        clean
        .groupby(
            [
                "adequacy_cutoff_pct",
                "response",
                "extractor",
                "model_type",
                "probability_level",
            ],
            as_index=False,
        )
        .agg(
            n_thresholds=("threshold_mg_dm3", "count"),
            median_threshold_mg_dm3=("threshold_mg_dm3", "median"),
            mean_threshold_mg_dm3=("threshold_mg_dm3", "mean"),
            min_threshold_mg_dm3=("threshold_mg_dm3", "min"),
            max_threshold_mg_dm3=("threshold_mg_dm3", "max"),
            q1_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.25)),
            q3_threshold_mg_dm3=("threshold_mg_dm3", lambda x: x.quantile(0.75)),
            auc=("auc", "median"),
            brier_score=("brier_score", "median"),
            slope=("slope", "median"),
            slope_p_value=("slope_p_value", "median"),
        )
    )

    summary["iqr_threshold_mg_dm3"] = (
        summary["q3_threshold_mg_dm3"] - summary["q1_threshold_mg_dm3"]
    )

    summary["cv_threshold"] = (
        clean
        .groupby(
            [
                "adequacy_cutoff_pct",
                "response",
                "extractor",
                "model_type",
                "probability_level",
            ]
        )["threshold_mg_dm3"]
        .agg(lambda x: np.std(x, ddof=1) / np.mean(x) if len(x) > 1 and np.mean(x) != 0 else np.nan)
        .to_numpy()
    )

    return summary.sort_values(
        [
            "adequacy_cutoff_pct",
            "response",
            "model_type",
            "probability_level",
            "auc",
            "cv_threshold",
        ],
        ascending=[True, True, True, True, False, True],
    )


def print_primary_results(model_results: pd.DataFrame, threshold_summary: pd.DataFrame) -> None:
    print("\n=== Logistic model performance at 90% relative response ===")

    models_90 = model_results[
        (model_results["adequacy_cutoff_pct"] == 90.0)
        & (model_results["response"].isin(["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]))
    ].copy()

    for response in ["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]:
        print(f"\nResponse: {response}")
        subset = models_90[models_90["response"] == response].copy()
        print(
            subset[
                [
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
            ]
            .sort_values(["model_type", "auc"], ascending=[True, False])
            .to_string(index=False)
        )

    print("\n=== Probability thresholds at 90% relative response, soil-fixed logistic, P = 0.50 ===")

    thresh = threshold_summary[
        (threshold_summary["adequacy_cutoff_pct"] == 90.0)
        & (threshold_summary["model_type"] == "soil_fixed_logistic")
        & (threshold_summary["probability_level"] == 0.50)
        & (threshold_summary["response"].isin(["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]))
    ].copy()

    for response in ["P_uptake_total_rel_pct", "dry_matter_total_rel_pct"]:
        print(f"\nResponse: {response}")
        subset = thresh[thresh["response"] == response].copy()
        print(
            subset[
                [
                    "extractor",
                    "n_thresholds",
                    "median_threshold_mg_dm3",
                    "min_threshold_mg_dm3",
                    "max_threshold_mg_dm3",
                    "iqr_threshold_mg_dm3",
                    "cv_threshold",
                    "auc",
                    "brier_score",
                    "slope_p_value",
                ]
            ]
            .sort_values(["auc", "cv_threshold"], ascending=[False, True])
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

    model_rows = []
    threshold_rows = []

    for cutoff in ADEQUACY_CUTOFFS:
        for response_col in RESPONSE_COLUMNS:
            for extractor, group in df.groupby("extractor"):
                for model_type in ["pooled_logistic", "soil_fixed_logistic"]:
                    model_row, thresholds = analyze_model(group, response_col, cutoff, extractor, model_type)

                    if model_row is not None:
                        model_rows.append(model_row)
                    threshold_rows.extend(thresholds)

    model_results = pd.DataFrame(model_rows)
    thresholds = pd.DataFrame(threshold_rows)
    threshold_summary = summarize_thresholds(thresholds)

    model_results.to_csv(MODEL_RESULTS_PATH, index=False)
    thresholds.to_csv(PROB_THRESHOLDS_PATH, index=False)
    threshold_summary.to_csv(PROB_THRESHOLDS_SUMMARY_PATH, index=False)

    print("Probabilistic relative response threshold analysis finished.")
    print(f"Model results: {MODEL_RESULTS_PATH}")
    print(f"Probability thresholds: {PROB_THRESHOLDS_PATH}")
    print(f"Probability threshold summary: {PROB_THRESHOLDS_SUMMARY_PATH}")

    print_primary_results(model_results, threshold_summary)


if __name__ == "__main__":
    main()
