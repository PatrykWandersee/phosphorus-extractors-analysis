from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"

TABLES_DIR = BASE_DIR / "tables/private"
MODEL_DIAGNOSTICS_PATH = TABLES_DIR / "plant_response_model_diagnostics.csv"
MODEL_RECOMMENDATIONS_PATH = TABLES_DIR / "plant_response_model_recommendations.csv"
RSTUDENT_OUTLIERS_PATH = TABLES_DIR / "plant_response_rstudent_outliers.csv"


RESPONSE_COLUMNS = [
    "dry_matter_cut1_g",
    "dry_matter_cut2_g",
    "dry_matter_total_g",
    "P_uptake_cut1_mg_pot",
    "P_uptake_cut2_mg_pot",
    "P_uptake_total_mg_pot",
]

RSTUDENT_THRESHOLD = 3.0


def fit_ols_model(group: pd.DataFrame, response: str, model_type: str):
    cols = ["sample_code", "p_dose_mg_dm3", response]
    clean = group[cols].dropna().copy()

    if len(clean) < 6:
        return None, clean

    x = clean["p_dose_mg_dm3"].astype(float)
    y = clean[response].astype(float)

    if model_type == "linear":
        X = pd.DataFrame({
            "intercept": 1.0,
            "dose": x,
        })
    elif model_type == "quadratic":
        X = pd.DataFrame({
            "intercept": 1.0,
            "dose": x,
            "dose_sq": x ** 2,
        })
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = sm.OLS(y, X).fit()

    return model, clean


def extract_model_diagnostics(soil_id, soil_label, response, model_type, model, clean):
    if model is None:
        return None, []

    dose_min = float(clean["p_dose_mg_dm3"].min())
    dose_max = float(clean["p_dose_mg_dm3"].max())

    params = model.params
    pvalues = model.pvalues

    intercept = params.get("intercept", np.nan)
    linear_coef = params.get("dose", np.nan)
    quadratic_coef = params.get("dose_sq", np.nan)

    linear_p = pvalues.get("dose", np.nan)
    quadratic_p = pvalues.get("dose_sq", np.nan)

    vertex = np.nan
    quadratic_plausible = False

    if model_type == "quadratic" and pd.notna(quadratic_coef) and quadratic_coef != 0:
        vertex = -linear_coef / (2 * quadratic_coef)
        quadratic_plausible = (
            quadratic_coef < 0
            and dose_min <= vertex <= dose_max
        )

    influence = model.get_influence()
    rstudent = influence.resid_studentized_external

    outlier_rows = []
    for sample_code, dose, value, rs in zip(
        clean["sample_code"],
        clean["p_dose_mg_dm3"],
        clean[response],
        rstudent,
    ):
        if abs(rs) > RSTUDENT_THRESHOLD:
            outlier_rows.append({
                "soil_id": soil_id,
                "soil_label": soil_label,
                "response": response,
                "model_type": model_type,
                "sample_code": sample_code,
                "p_dose_mg_dm3": dose,
                "observed_value": value,
                "rstudent": rs,
            })

    diagnostics = {
        "soil_id": soil_id,
        "soil_label": soil_label,
        "response": response,
        "model_type": model_type,
        "n": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "intercept": intercept,
        "linear_coef": linear_coef,
        "quadratic_coef": quadratic_coef,
        "linear_p": linear_p,
        "quadratic_p": quadratic_p,
        "dose_min": dose_min,
        "dose_max": dose_max,
        "vertex_mg_dm3": vertex,
        "quadratic_plausible": quadratic_plausible,
        "n_rstudent_outliers": len(outlier_rows),
        "max_abs_rstudent": float(np.nanmax(np.abs(rstudent))),
    }

    return diagnostics, outlier_rows


def classify_response(linear_row, quadratic_row):
    if linear_row is None and quadratic_row is None:
        return {
            "selected_model": "none",
            "model_status": "insufficient_data",
            "selection_reason": "fewer_than_minimum_observations",
        }

    linear_ok = (
        linear_row is not None
        and linear_row["linear_coef"] > 0
        and linear_row["linear_p"] < 0.05
    )

    quadratic_ok = (
        quadratic_row is not None
        and bool(quadratic_row["quadratic_plausible"])
        and quadratic_row["quadratic_p"] < 0.05
    )

    if quadratic_ok and linear_row is not None:
        delta_aic = linear_row["aic"] - quadratic_row["aic"]
        if delta_aic > 2:
            return {
                "selected_model": "quadratic",
                "model_status": "supported",
                "selection_reason": "plausible_quadratic_with_aic_improvement",
            }

    if linear_ok:
        return {
            "selected_model": "linear",
            "model_status": "supported",
            "selection_reason": "positive_significant_linear_response",
        }

    if linear_row is not None and linear_row["linear_coef"] > 0:
        return {
            "selected_model": "linear",
            "model_status": "weak_positive",
            "selection_reason": "positive_but_not_significant_linear_response",
        }

    return {
        "selected_model": "none",
        "model_status": "not_supported",
        "selection_reason": "no_positive_supported_response",
    }


def build_recommendations(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (soil_id, soil_label, response), group in diagnostics.groupby(["soil_id", "soil_label", "response"]):
        linear_match = group[group["model_type"] == "linear"]
        quadratic_match = group[group["model_type"] == "quadratic"]

        linear_row = linear_match.iloc[0].to_dict() if len(linear_match) else None
        quadratic_row = quadratic_match.iloc[0].to_dict() if len(quadratic_match) else None

        classification = classify_response(linear_row, quadratic_row)

        selected_row = None
        if classification["selected_model"] == "linear":
            selected_row = linear_row
        elif classification["selected_model"] == "quadratic":
            selected_row = quadratic_row

        rows.append({
            "soil_id": soil_id,
            "soil_label": soil_label,
            "response": response,
            **classification,
            "selected_r_squared": selected_row["r_squared"] if selected_row else np.nan,
            "selected_adj_r_squared": selected_row["adj_r_squared"] if selected_row else np.nan,
            "selected_aic": selected_row["aic"] if selected_row else np.nan,
            "selected_linear_p": selected_row["linear_p"] if selected_row else np.nan,
            "selected_quadratic_p": selected_row["quadratic_p"] if selected_row else np.nan,
            "selected_n_rstudent_outliers": selected_row["n_rstudent_outliers"] if selected_row else np.nan,
            "selected_max_abs_rstudent": selected_row["max_abs_rstudent"] if selected_row else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSIS_DATASET_PATH)

    diagnostics_rows = []
    outlier_rows = []

    for (soil_id, soil_label), group in df.groupby(["soil_id", "soil_label"]):
        for response in RESPONSE_COLUMNS:
            for model_type in ["linear", "quadratic"]:
                model, clean = fit_ols_model(group, response, model_type)
                diagnostics, outliers = extract_model_diagnostics(
                    soil_id,
                    soil_label,
                    response,
                    model_type,
                    model,
                    clean,
                )

                if diagnostics is not None:
                    diagnostics_rows.append(diagnostics)
                outlier_rows.extend(outliers)

    diagnostics = pd.DataFrame(diagnostics_rows)
    recommendations = build_recommendations(diagnostics)
    outliers = pd.DataFrame(outlier_rows)

    diagnostics.to_csv(MODEL_DIAGNOSTICS_PATH, index=False)
    recommendations.to_csv(MODEL_RECOMMENDATIONS_PATH, index=False)
    outliers.to_csv(RSTUDENT_OUTLIERS_PATH, index=False)

    print("Plant response model diagnostics finished.")
    print(f"Model diagnostics: {MODEL_DIAGNOSTICS_PATH}")
    print(f"Model recommendations: {MODEL_RECOMMENDATIONS_PATH}")
    print(f"RStudent outliers: {RSTUDENT_OUTLIERS_PATH}")

    print("\nModel status counts:")
    print(recommendations["model_status"].value_counts().to_string())

    for response in ["dry_matter_total_g", "P_uptake_total_mg_pot", "dry_matter_cut1_g", "P_uptake_cut1_mg_pot"]:
        print(f"\nRecommendations for {response}:")
        subset = recommendations[recommendations["response"] == response].copy()
        print(
            subset[[
                "soil_label",
                "selected_model",
                "model_status",
                "selected_r_squared",
                "selected_linear_p",
                "selected_quadratic_p",
                "selected_n_rstudent_outliers",
                "selection_reason",
            ]]
            .sort_values("soil_label")
            .to_string(index=False)
        )

    print(f"\nTotal RStudent outlier flags: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers.to_string(index=False))


if __name__ == "__main__":
    main()
