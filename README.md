# Phosphorus Extractors Analysis

A compact, reproducible data-analysis project inspired by the article **"Soil properties and pH of the extractors influence extraction and availability P in alkaline soils from the São Francisco Valley, Brazil"**.

## What this project does

This repository uses a small sample dataset derived from the soil characterization table in the paper to explore:

- how phosphorus values differ across extractors
- how soil pH, exchangeable calcium, clay, and P-rem relate to extractor behavior
- why **Mehlich-3** stands out as a more stable option in alkaline soils

## Repository structure

```text
phosphorus-extractors-analysis-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample_phosphorus_data.csv
├── notebooks/
│   ├── phosphorus_extractors_analysis.ipynb
│   └── phosphorus_extractors_analysis.py
└── figures/
```

## Dataset

The sample dataset includes six soils used in the paper:

- RY
- RQ
- PVA1
- PVA2
- LVA
- V

Variables included:

- soil pH
- exchangeable Ca
- clay
- P-rem
- phosphorus extracted by Mehlich-1, Mehlich-3, AER, and Olsen

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python notebooks/phosphorus_extractors_analysis.py
```

Or open the notebook in Jupyter / Google Colab.

## Key takeaway

The paper concludes that **exchangeable calcium and soil pH influence phosphorus extraction more than clay and P-rem**, and that **Mehlich-3 showed high performance in predicting P availability in alkaline environments**.

## Why this repo is useful

This is not meant to reproduce the full paper. It is a portfolio-style project designed to show:

- clean project organization
- scientific data handling in Python
- exploratory analysis
- concise interpretation of results
- a reproducible workflow that can later be expanded into regressions, benchmarking, and richer visualization
