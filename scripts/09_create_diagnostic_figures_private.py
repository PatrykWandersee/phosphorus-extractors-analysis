from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

ANALYSIS_DATASET_PATH = BASE_DIR / "data/private/processed/experiment_analysis_dataset.csv"
OUTLIERS_PATH = BASE_DIR / "tables/private/plant_response_rstudent_outliers.csv"

FIGURES_DIR = BASE_DIR / "figures/private"

EXTRACTOR_COLUMNS = {
    "P_Mehlich1_mg_dm3": "Mehlich-1",
    "P_Mehlich3_mg_dm3": "Mehlich-3",
    "P_resin_mg_dm3": "Resin",
    "P_Olsen_mg_dm3": "Olsen",
}

RESPONSE_LABELS = {
    "dry_matter_total_g": "Total dry matter (g pot$^{-1}$)",
    "P_uptake_total_mg_pot": "Total P uptake (mg pot$^{-1}$)",
    "dry_matter_cut1_g": "Dry matter, first cut (g pot$^{-1}$)",
    "P_uptake_cut1_mg_pot": "P uptake, first cut (mg pot$^{-1}$)",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ANALYSIS_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found: {ANALYSIS_DATASET_PATH}\n"
            "Run scripts/01_prepare_analysis_dataset.py first."
        )

    df = pd.read_csv(ANALYSIS_DATASET_PATH)
    df["sample_code"] = df["sample_code"].astype(str)

    if OUTLIERS_PATH.exists() and OUTLIERS_PATH.stat().st_size > 0:
        outliers = pd.read_csv(OUTLIERS_PATH)
        outliers["sample_code"] = outliers["sample_code"].astype(str)
    else:
        outliers = pd.DataFrame()

    return df, outliers


def add_linear_trend(ax, x, y):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(clean) < 3:
        return

    coef = np.polyfit(clean["x"], clean["y"], deg=1)
    x_grid = np.linspace(clean["x"].min(), clean["x"].max(), 100)
    y_grid = coef[0] * x_grid + coef[1]

    ax.plot(x_grid, y_grid, linewidth=1)


def mark_outliers(ax, df_soil, outliers, response):
    if outliers.empty:
        return

    flagged = outliers[outliers["response"] == response]["sample_code"].unique()
    if len(flagged) == 0:
        return

    flagged_df = df_soil[df_soil["sample_code"].isin(flagged)]

    for _, row in flagged_df.iterrows():
        x = row["p_dose_mg_dm3"]
        y = row[response]

        if pd.notna(x) and pd.notna(y):
            ax.scatter([x], [y], marker="x", s=60)
            ax.annotate(
                row["sample_code"],
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )


def plot_response_by_soil(df, outliers, response):
    soils = (
        df[["soil_id", "soil_label"]]
        .drop_duplicates()
        .sort_values("soil_id")
        .to_records(index=False)
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (soil_id, soil_label) in zip(axes, soils):
        df_soil = df[df["soil_id"] == soil_id].copy()

        ax.scatter(df_soil["p_dose_mg_dm3"], df_soil[response], s=30)
        add_linear_trend(ax, df_soil["p_dose_mg_dm3"], df_soil[response])
        mark_outliers(ax, df_soil, outliers, response)

        ax.set_title(f"{soil_label}")
        ax.set_xlabel("P dose (mg dm$^{-3}$)")
        ax.set_ylabel(RESPONSE_LABELS[response])
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{RESPONSE_LABELS[response]} by soil and P dose")

    output_base = FIGURES_DIR / f"diagnostic_{response}_by_soil"
    fig.savefig(f"{output_base}.png", dpi=300)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def plot_extractor_recovery_by_soil(df):
    soils = (
        df[["soil_id", "soil_label"]]
        .drop_duplicates()
        .sort_values("soil_id")
        .to_records(index=False)
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (soil_id, soil_label) in zip(axes, soils):
        df_soil = df[df["soil_id"] == soil_id].copy()

        for col, label in EXTRACTOR_COLUMNS.items():
            ax.scatter(df_soil["p_dose_mg_dm3"], df_soil[col], s=20, label=label)
            add_linear_trend(ax, df_soil["p_dose_mg_dm3"], df_soil[col])

        ax.set_title(f"{soil_label}")
        ax.set_xlabel("P dose (mg dm$^{-3}$)")
        ax.set_ylabel("Extracted P (mg dm$^{-3}$)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("Extractor recovery curves by soil")

    output_base = FIGURES_DIR / "diagnostic_extractor_recovery_curves_by_soil"
    fig.savefig(f"{output_base}.png", dpi=300)
    fig.savefig(f"{output_base}.pdf")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, outliers = load_data()

    for response in RESPONSE_LABELS:
        plot_response_by_soil(df, outliers, response)

    plot_extractor_recovery_by_soil(df)

    print("Diagnostic figures created:")
    for path in sorted(FIGURES_DIR.glob("diagnostic_*")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
