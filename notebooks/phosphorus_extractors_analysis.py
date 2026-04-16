import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sample_phosphorus_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

extractor_cols = ["P_Mehlich1_mg_dm3", "P_Mehlich3_mg_dm3", "P_AER_mg_dm3", "P_Olsen_mg_dm3"]
soil_cols = ["pH_H2O", "exchangeable_Ca_mmolc_dm3", "clay_g_kg", "P_rem_mg_L"]

print("\nDataset preview:")
print(df)

print("\nDescriptive statistics:")
print(df.describe().round(2))

corr = df[soil_cols + extractor_cols].corr().round(2)
print("\nCorrelation matrix:")
print(corr)

# 1. Extractor comparison by soil
ax = df.set_index("soil")[extractor_cols].plot(kind="bar", figsize=(10, 6))
ax.set_title("Phosphorus extracted by method across soils")
ax.set_xlabel("Soil")
ax.set_ylabel("P extracted (mg dm$^{-3}$)")
ax.legend(["Mehlich-1", "Mehlich-3", "AER", "Olsen"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "extractors_by_soil.png"), dpi=200)
plt.close()

# 2. Mehlich-3 vs pH
plt.figure(figsize=(8, 5))
plt.scatter(df["pH_H2O"], df["P_Mehlich3_mg_dm3"])
for _, row in df.iterrows():
    plt.annotate(row["soil"], (row["pH_H2O"], row["P_Mehlich3_mg_dm3"]), xytext=(5, 5), textcoords="offset points")
plt.title("Mehlich-3 extraction vs soil pH")
plt.xlabel("Soil pH")
plt.ylabel("Mehlich-3 (mg dm$^{-3}$)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mehlich3_vs_pH.png"), dpi=200)
plt.close()

# 3. Mehlich-1 vs exchangeable Ca
plt.figure(figsize=(8, 5))
plt.scatter(df["exchangeable_Ca_mmolc_dm3"], df["P_Mehlich1_mg_dm3"])
for _, row in df.iterrows():
    plt.annotate(row["soil"], (row["exchangeable_Ca_mmolc_dm3"], row["P_Mehlich1_mg_dm3"]), xytext=(5, 5), textcoords="offset points")
plt.title("Mehlich-1 extraction vs exchangeable Ca")
plt.xlabel("Exchangeable Ca (mmolc dm$^{-3}$)")
plt.ylabel("Mehlich-1 (mg dm$^{-3}$)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "mehlich1_vs_Ca.png"), dpi=200)
plt.close()

# 4. Mehlich-1 / Mehlich-3 ratio
df["M1_to_M3_ratio"] = df["P_Mehlich1_mg_dm3"] / df["P_Mehlich3_mg_dm3"]
plt.figure(figsize=(8, 5))
plt.bar(df["soil"], df["M1_to_M3_ratio"])
plt.title("Mehlich-1 / Mehlich-3 ratio by soil")
plt.xlabel("Soil")
plt.ylabel("Ratio")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "m1_to_m3_ratio.png"), dpi=200)
plt.close()

print("\nInterpretation:")
print("- Mehlich-1 varies strongly across soils and can behave inconsistently in alkaline conditions.")
print("- Mehlich-3 remains comparatively stable across most soils in this small dataset.")
print("- Soil V is a clear outlier because it combines high pH, high exchangeable Ca, and high clay.")
print("- This project is a compact exploratory workflow, suitable for expansion into regression or benchmarking analyses.")
print(f"\nFigures saved in: {FIG_DIR}")
