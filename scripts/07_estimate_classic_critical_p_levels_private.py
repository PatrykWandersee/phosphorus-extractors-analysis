from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"
RECOVERY_RATES_PATH = BASE_DIR / "tables/private/extractor_recovery_rates_by_soil.csv"

TABLES_DIR = BASE_DIR / "tables/private"

PLANT_MODELS_PATH = TABLES_DIR / "plant_response_models_by_soil.csv"
RECOMMENDED_RATES_PATH = TABLES_DIR / "recommended_p_rates_by_soil.csv"
CLASSIC_CRITICAL_LEVELS_PATH = TABLES_DIR / "classic_critical_p_levels_by_extractor.csv"
CLASSIC_CRITICAL_SUMMARY_PATH = TABLES_DIR / "classic_critical_p_levels_summary.csv"


RESPONSE_COLUMNS = [
    "dry_matter_cut1_g",
    "dry_matter_cut2_g",
    "dry_matter_total_g",
]

EXTRACTOR_COLUMNS = {
    "P_Mehlich1_mg_dm3": "Mehlich-1",
    "P_Mehlich3_mg_dm3": "Mehlich-3",
    "P_resin_mg_dm3": "Resin",
    "P_Olsen_mg_dm3": "Olsen",
}


def fit_model(x: np.ndarray, y: np.ndarray, model_type: str) -> dict:
    if model_type == "linear":
        X = np.column_stack([np.ones_like(x), x])
        k = 2
    elif model_type == "quadratic":
        X = np.column_stack([np.ones_like(x), x, x**2])
        k = 3
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat

    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y))**2))
    n = len(y)

    r_squared = 1 - rss / tss if tss > 0 else np.nan

    if rss <= 0:
        aic = np.nan
    else:
        aic = n * np.log(rss / n) + 2 * k

    return {
        "model_type": model_type,
        "n": n,
        "intercept": beta[0],
        "linear_coef": beta[1],
        "quadratic_coef": beta[2] if model_type == "quadratic" else np.nan,
        "rss": rss,
        "r_squared": r_squared,
        "aic": aic,
    }


def predict_response(x: float, fit: dict) -> float:
    y = fit["intercept"] + fit["linear_coef"] * x
    if fit["model_type"] == "quadratic":
        y += fit["quadratic_coef"] * x**2
    return float(y)


def choose_response_model(linear_fit: dict, quadratic_fit: dict, max_dose: float) -> dict:
    q = quadratic_fit.copy()
    l = linear_fit.copy()

    q_valid = (
        np.isfinite(q["quadratic_coef"])
        and q["quadratic_coef"] < 0
    )

    if q_valid:
        vertex = -q["linear_coef"] / (2 * q["quadratic_coef"])
        q_valid = 0 <= vertex <= max_dose

    # Prefer quadratic only if it is biologically plausible and improves AIC.
    if q_valid and np.isfinite(q["aic"]) and np.isfinite(l["aic"]) and q["aic"] < (l["aic"] - 2):
        q["selected"] = True
        q["selection_reason"] = "quadratic_valid_and_aic_improvement"
        return q

    l["selected"] = True
    l["selection_reason"] = "linear_selected"
    return l


def estimate_recommended_rate(fit: dict, min_dose: float, max_dose: float) -> dict:
    model_type = fit["model_type"]

    if model_type == "quadratic":
        a = fit["intercept"]
        b = fit["linear_coef"]
        c = fit["quadratic_coef"]

        if not np.isfinite(c) or c >= 0:
            return {
                "max_response_dose_mg_dm3": np.nan,
                "max_response": np.nan,
                "target_90pct_response": np.nan,
                "recommended_p_rate_mg_dm3": np.nan,
                "recommended_rate_method": "invalid_quadratic",
            }

        x_max = -b / (2 * c)
        x_max = float(np.clip(x_max, min_dose, max_dose))
        y_max = predict_response(x_max, fit)
        target = 0.9 * y_max

        roots = np.roots([c, b, a - target])
        real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-8]
        candidate_roots = [r for r in real_roots if min_dose <= r <= x_max]

        if not candidate_roots:
            recommended = np.nan
        else:
            recommended = min(candidate_roots)

        return {
            "max_response_dose_mg_dm3": x_max,
            "max_response": y_max,
            "target_90pct_response": target,
            "recommended_p_rate_mg_dm3": recommended,
            "recommended_rate_method": "quadratic_90pct_of_predicted_maximum",
        }

    if model_type == "linear":
        a = fit["intercept"]
        b = fit["linear_coef"]

        if b <= 0:
            return {
                "max_response_dose_mg_dm3": min_dose,
                "max_response": predict_response(min_dose, fit),
                "target_90pct_response": np.nan,
                "recommended_p_rate_mg_dm3": np.nan,
                "recommended_rate_method": "linear_nonpositive_slope_no_rate_estimated",
            }

        x_max = max_dose
        y_max = predict_response(x_max, fit)
        target = 0.9 * y_max
        recommended = (target - a) / b
        recommended = float(np.clip(recommended, min_dose, max_dose))

        return {
            "max_response_dose_mg_dm3": x_max,
            "max_response": y_max,
            "target_90pct_response": target,
            "recommended_p_rate_mg_dm3": recommended,
            "recommended_rate_method": "linear_90pct_of_predicted_response_at_max_dose",
        }

    raise ValueError(f"Unknown model type: {model_type}")


def fit_plant_response_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    rate_rows = []

    for (soil_id, soil_label), soil_df in df.groupby(["soil_id", "soil_label"]):
        min_dose = float(soil_df["p_dose_mg_dm3"].min())
        max_dose = float(soil_df["p_dose_mg_dm3"].max())

        for response_col in RESPONSE_COLUMNS:
            clean = soil_df[["p_dose_mg_dm3", response_col]].dropna().copy()

            if len(clean) < 6:
                continue

            x = clean["p_dose_mg_dm3"].to_numpy(dtype=float)
            y = clean[response_col].to_numpy(dtype=float)

            linear_fit = fit_model(x, y, "linear")
            quadratic_fit = fit_model(x, y, "quadratic")

            for fit in [linear_fit, quadratic_fit]:
                model_rows.append({
                    "soil_id": soil_id,
                    "soil_label": soil_label,
                    "response": response_col,
                    **fit,
                })

            selected = choose_response_model(linear_fit, quadratic_fit, max_dose)
            rate = estimate_recommended_rate(selected, min_dose, max_dose)

            rate_rows.append({
                "soil_id": soil_id,
                "soil_label": soil_label,
                "response": response_col,
                "selected_model": selected["model_type"],
                "selection_reason": selected["selection_reason"],
                "n": selected["n"],
                "r_squared": selected["r_squared"],
                "aic": selected["aic"],
                "intercept": selected["intercept"],
                "linear_coef": selected["linear_coef"],
                "quadratic_coef": selected["quadratic_coef"],
                **rate,
            })

    return pd.DataFrame(model_rows), pd.DataFrame(rate_rows)


def estimate_classic_critical_levels(rates: pd.DataFrame, recovery: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, rate_row in rates.iterrows():
        soil_id = rate_row["soil_id"]
        soil_label = rate_row["soil_label"]
        response = rate_row["response"]
        dr = rate_row["recommended_p_rate_mg_dm3"]

        recovery_subset = recovery[
            (recovery["soil_id"] == soil_id)
            & (recovery["soil_label"] == soil_label)
        ]

        for _, rec_row in recovery_subset.iterrows():
            if pd.isna(dr):
                critical_p = np.nan
            else:
                critical_p = rec_row["intercept"] + rec_row["recovery_rate_slope"] * dr

            rows.append({
                "soil_id": soil_id,
                "soil_label": soil_label,
                "response": response,
                "extractor": rec_row["extractor"],
                "recommended_p_rate_mg_dm3": dr,
                "critical_P_mg_dm3": critical_p,
                "selected_plant_model": rate_row["selected_model"],
                "plant_model_r_squared": rate_row["r_squared"],
                "extractor_recovery_rate": rec_row["recovery_rate_slope"],
                "extractor_model_r_squared": rec_row["r_squared"],
            })

    return pd.DataFrame(rows)


def summarize_critical_levels(critical: pd.DataFrame) -> pd.DataFrame:
    clean = critical.dropna(subset=["critical_P_mg_dm3"]).copy()

    summary = (
        clean
        .groupby(["response", "extractor"], as_index=False)
        .agg(
            n_soils=("soil_id", "count"),
            median_critical_P_mg_dm3=("critical_P_mg_dm3", "median"),
            mean_critical_P_mg_dm3=("critical_P_mg_dm3", "mean"),
            min_critical_P_mg_dm3=("critical_P_mg_dm3", "min"),
            max_critical_P_mg_dm3=("critical_P_mg_dm3", "max"),
            median_recommended_p_rate_mg_dm3=("recommended_p_rate_mg_dm3", "median"),
        )
    )

    return summary.sort_values(["response", "median_critical_P_mg_dm3"])


def main() -> None:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}"
        )

    if not RECOVERY_RATES_PATH.exists():
        raise FileNotFoundError(
            f"Recovery-rate table not found: {RECOVERY_RATES_PATH}\n"
            "Run scripts/06_estimate_extractor_recovery_rates_private.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DATASET_PATH)
    recovery = pd.read_csv(RECOVERY_RATES_PATH)

    plant_models, recommended_rates = fit_plant_response_models(df)
    critical_levels = estimate_classic_critical_levels(recommended_rates, recovery)
    critical_summary = summarize_critical_levels(critical_levels)

    plant_models.to_csv(PLANT_MODELS_PATH, index=False)
    recommended_rates.to_csv(RECOMMENDED_RATES_PATH, index=False)
    critical_levels.to_csv(CLASSIC_CRITICAL_LEVELS_PATH, index=False)
    critical_summary.to_csv(CLASSIC_CRITICAL_SUMMARY_PATH, index=False)

    print("Classic critical P level estimation finished.")
    print(f"Plant response models: {PLANT_MODELS_PATH}")
    print(f"Recommended P rates: {RECOMMENDED_RATES_PATH}")
    print(f"Critical P levels: {CLASSIC_CRITICAL_LEVELS_PATH}")
    print(f"Critical P summary: {CLASSIC_CRITICAL_SUMMARY_PATH}")

    print("\nRecommended P rates by soil and response:")
    print(
        recommended_rates[[
            "soil_label",
            "response",
            "selected_model",
            "r_squared",
            "recommended_p_rate_mg_dm3",
            "recommended_rate_method",
        ]]
        .sort_values(["response", "soil_label"])
        .to_string(index=False)
    )

    for response in RESPONSE_COLUMNS:
        print(f"\nCritical P summary for {response}:")
        subset = critical_summary[critical_summary["response"] == response].copy()
        print(
            subset[[
                "extractor",
                "n_soils",
                "median_critical_P_mg_dm3",
                "min_critical_P_mg_dm3",
                "max_critical_P_mg_dm3",
            ]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
