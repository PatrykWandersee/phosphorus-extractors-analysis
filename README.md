# Phosphorus Extractors Analysis

Reproducible workflow for evaluating phosphorus extractors in alkaline soils from the Brazilian semiarid region.

This repository documents a scientific data-analysis pipeline derived from a greenhouse experiment with *Brachiaria decumbens* grown in soils with contrasting chemical and physical properties. The workflow compares four soil phosphorus extractors: Mehlich-1, Mehlich-3, anion exchange resin, and Olsen.

The original experimental dataset is not included in this repository because it is associated with thesis-derived analyses and potential future publications. Private data, tables, figures, and synthesis reports are stored locally in ignored directories.

## Scientific objective

The main goal is to evaluate how well different soil P extractors reflect plant response under alkaline soil conditions.

The workflow distinguishes between extractor recovery, biological prediction, soil-specific behavior, and diagnostic threshold behavior.

## Current analysis status

This repository currently contains a reproducible workflow for exploring phosphorus extractor performance, plant response variables, and diagnostic threshold behavior in alkaline soils.

The original experimental dataset and detailed numerical outputs are kept private. Public-facing scripts are organized to document the analytical workflow, while private data, tables, figures, and synthesis reports remain excluded from version control.

Recent private analyses include data validation, extractor-response relationships, extractor recovery-rate estimation, plant response model diagnostics, influence screening, soil fixed-effect and interaction models, relative response threshold screening, probabilistic adequacy modeling, bootstrap and leave-one-soil-out threshold validation, and response-specific outlier sensitivity analysis.

The current internal interpretation is methodological rather than prescriptive: relative plant P uptake appears more informative than dry matter alone for diagnostic threshold reassessment, but threshold transferability among contrasting soils remains limited. Therefore, this workflow is being treated as an exploratory and reproducible analysis framework rather than as a source of universal critical P values.

## Repository structure

Main folders:

- data/sample: public sample data, when available.
- data/private: private data; ignored and not tracked.
- figures/private: private figures; ignored and not tracked.
- manuscript/private: private synthesis reports; ignored and not tracked.
- notebooks: notebooks and exported notebook scripts.
- scripts: analysis scripts.
- tables/private: private tables; ignored and not tracked.

## Privacy note

This repository does not publish the original experimental data. Files under data/private, tables/private, figures/private, and manuscript/private are intentionally excluded from version control.

The scripts document the analytical logic, but full reproduction of the private results requires access to the original local dataset.

## Planned additions

Planned public-facing additions include a synthetic demonstration dataset, a reproducible example notebook, publication-style figures using synthetic data, and documentation explaining the difference between extractor recovery, biological prediction, and diagnostic threshold behavior.

## Author

Patryk Ramon Graciano Rosa Wandersee  
Agronomist | MSc and PhD in Soil Science  
Research interests: soil fertility, phosphorus dynamics, soil quality, plant-soil relationships, and applied data analysis.
