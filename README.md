# Phosphorus Extractors Analysis

Reproducible workflow for evaluating phosphorus extractors in alkaline soils from the Brazilian semiarid region.

This repository documents a scientific data-analysis pipeline derived from a greenhouse experiment with *Brachiaria decumbens* grown in soils with contrasting chemical and physical properties. The workflow compares four soil phosphorus extractors: Mehlich-1, Mehlich-3, anion exchange resin, and Olsen.

The original experimental dataset is not included in this repository because it is associated with thesis-derived analyses and potential future publications. Private data, tables, and figures are stored locally in ignored directories.

## Scientific objective

The main goal is to evaluate how well different soil P extractors reflect plant response under alkaline soil conditions.

The workflow distinguishes between:

1. **Extractor recovery** — how much P each method extracts as applied P increases.
2. **Biological prediction** — how well extracted P predicts plant P uptake and dry matter production.
3. **Soil-specific behavior** — whether extractor-response relationships vary among soils.

## Current working interpretation

Private analyses indicate that Mehlich-1 shows the greatest overall P recovery. However, higher recovery does not necessarily imply better prediction of plant response.

When soil effects are controlled, Mehlich-3 is the most consistent predictor of plant P uptake, especially total plant P uptake. Resin also performs well in sensitivity analyses, while Olsen shows strong within-soil correlations and relevant soil-specific behavior.

For this reason, plant P uptake is treated as the primary biological response in the current workflow. Dry matter production is retained as a complementary agronomic response.

## Repository structure

```text
phosphorus-extractors-analysis/
├── data/
│   ├── sample/
│   └── private/      # ignored; not tracked
├── figures/
│   └── private/      # ignored; not tracked
├── notebooks/
├── scripts/
├── tables/
│   └── private/      # ignored; not tracked
├── requirements.txt
└── README.md
```

## Methods represented

The private workflow currently includes:

- data validation and cleaning;
- Pearson and Spearman correlations;
- within-soil correlation analysis;
- extractor recovery-rate estimation;
- plant response model diagnostics;
- RStudent-based influence screening;
- soil fixed-effect models;
- soil-specific interaction models;
- sensitivity analysis with and without influential observations;
- private diagnostic figures and synthesis reports.

## Privacy note

This repository does not publish the original experimental data. Files under `data/private/`, `tables/private/`, and `figures/private/` are intentionally excluded from version control.

The scripts document the analytical logic, but full reproduction of the private results requires access to the original local dataset.

## Planned additions

Planned public-facing additions include:

- a synthetic demonstration dataset;
- a reproducible example notebook;
- publication-style figures using synthetic data;
- documentation explaining the difference between extractor recovery and biological prediction.

## Author

**Patryk Ramon Graciano Rosa Wandersee**  
Agronomist | MSc and PhD in Soil Science  
Research interests: soil fertility, phosphorus dynamics, soil quality, plant-soil relationships, and applied data analysis.
