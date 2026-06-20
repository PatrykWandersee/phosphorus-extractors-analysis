from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data/sample/synthetic_phosphorus_experiment.csv"

rng = np.random.default_rng(42)

soils = [
    {"soil_id": 1, "soil_label": "RY", "response_factor": 0.75, "buffering": 0.85},
    {"soil_id": 2, "soil_label": "RQ", "response_factor": 0.90, "buffering": 0.95},
    {"soil_id": 3, "soil_label": "PVA1", "response_factor": 1.05, "buffering": 1.05},
    {"soil_id": 4, "soil_label": "PVA2", "response_factor": 1.10, "buffering": 1.10},
    {"soil_id": 5, "soil_label": "LVA", "response_factor": 1.20, "buffering": 1.15},
    {"soil_id": 6, "soil_label": "V", "response_factor": 0.55, "buffering": 0.55},
]

relative_levels = [0, 10, 20, 40, 70, 100]

rows = []

for soil in soils:
    for level_code, level_pct in enumerate(relative_levels):
        max_dose = 770 if soil["soil_label"] == "V" else 370
        p_dose = max_dose * level_pct / 100

        for replicate in range(1, 5):
            sample_code = int(f"{soil['soil_id']}{level_code}{replicate}")

            # Synthetic extractor values. These are intentionally simulated and do not
            # reproduce the original experimental dataset.
            mehlich1 = 35 + 1.25 * p_dose * soil["buffering"] + rng.normal(0, 45)
            mehlich3 = 45 + 0.85 * p_dose * soil["buffering"] + rng.normal(0, 30)
            resin = 20 + 0.55 * p_dose * soil["buffering"] + rng.normal(0, 22)
            olsen = 15 + 0.38 * p_dose * soil["buffering"] + rng.normal(0, 14)

            mehlich1 = max(mehlich1, 0)
            mehlich3 = max(mehlich3, 0)
            resin = max(resin, 0)
            olsen = max(olsen, 0)

            response_curve = 1 - np.exp(-0.010 * p_dose)
            dry_matter_total = (
                8
                + 18 * soil["response_factor"] * response_curve
                + rng.normal(0, 1.8)
            )
            dry_matter_total = max(dry_matter_total, 0.5)

            dry_matter_cut1 = dry_matter_total * rng.normal(0.58, 0.04)
            dry_matter_cut2 = dry_matter_total - dry_matter_cut1

            plant_p_cut1 = 1.2 + 2.0 * response_curve + rng.normal(0, 0.20)
            plant_p_cut2 = 1.0 + 1.6 * response_curve + rng.normal(0, 0.18)

            plant_p_cut1 = max(plant_p_cut1, 0.2)
            plant_p_cut2 = max(plant_p_cut2, 0.2)

            p_uptake_cut1 = dry_matter_cut1 * plant_p_cut1
            p_uptake_cut2 = dry_matter_cut2 * plant_p_cut2
            p_uptake_total = p_uptake_cut1 + p_uptake_cut2

            rows.append({
                "sample_code": sample_code,
                "soil_id": soil["soil_id"],
                "soil_label": soil["soil_label"],
                "p_level_code": level_code,
                "p_level_percent": level_pct,
                "p_dose_mg_dm3": round(p_dose, 3),
                "replicate": replicate,
                "P_Mehlich1_mg_dm3": round(mehlich1, 3),
                "P_Mehlich3_mg_dm3": round(mehlich3, 3),
                "P_resin_mg_dm3": round(resin, 3),
                "P_Olsen_mg_dm3": round(olsen, 3),
                "dry_matter_cut1_g": round(dry_matter_cut1, 3),
                "dry_matter_cut2_g": round(dry_matter_cut2, 3),
                "dry_matter_total_g": round(dry_matter_total, 3),
                "plant_P_cut1_g_kg": round(plant_p_cut1, 3),
                "plant_P_cut2_g_kg": round(plant_p_cut2, 3),
                "P_uptake_cut1_mg_pot": round(p_uptake_cut1, 3),
                "P_uptake_cut2_mg_pot": round(p_uptake_cut2, 3),
                "P_uptake_total_mg_pot": round(p_uptake_total, 3),
            })

df = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Synthetic dataset written to: {OUTPUT_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
