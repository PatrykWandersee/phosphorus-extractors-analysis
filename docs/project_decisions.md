# Project decisions

## Project title

**Phosphorus extractors and critical P levels in alkaline soils from the Brazilian semiarid region**

## Source study

This project is based on the MSc experiment on phosphorus availability evaluated by different chemical extractors in alkaline soils from Pernambuco, Brazil.

## Experimental design

The greenhouse experiment used six soils and six phosphorus fertilization levels, with four replications, totaling 144 experimental units.

The experiment followed a randomized block design.

## Soil codes

The original sample code uses three digits:

```text
XYZ
```

where:

```text
X = soil ID
Y = phosphorus fertilization level code
Z = replicate
```

The soil ID ranges from 1 to 6.

The phosphorus level code represents the relative phosphorus level within each soil.

The replicate code ranges from 1 to 4.

## Phosphorus dose logic

Phosphorus fertilization rates were defined for each soil according to remaining phosphorus (P-rem), used as an indicator of phosphorus buffering capacity.

The relative phosphorus levels were:

```text
0 = 0%
1 = 10%
2 = 20%
3 = 40%
4 = 70%
5 = 100%
```

Because the maximum phosphorus rate was soil-specific, the same relative level does not necessarily represent the same absolute phosphorus dose across soils.

The Vertisol had a higher maximum phosphorus dose than the other soils because of its lower P-rem and greater phosphorus buffering capacity.

## Fertilizer source logic

Phosphorus was supplied primarily as KH2PO4.

When the potassium supplied by KH2PO4 reached the predefined potassium limit, additional phosphorus was supplied as NH4H2PO4.

KCl and (NH4)2SO4 were used to balance potassium and nitrogen inputs among treatments.

This nutrient-balancing logic must be preserved in the dose lookup table.

## Main raw data blocks

The project should preserve three main data blocks:

1. Experiment results
   One row per experimental unit, with soil, dose, replicate, block, extracted soil P after incubation, dry matter, plant P concentration, and plant P uptake.

2. Soil characterization
   Three characterization replicates per soil, including pH, salinity, soluble cations, exchangeable cations, CEC, ESP, organic carbon, P-rem, initial P by extractors, texture, water-dispersible clay, flocculation, bulk density, particle density, porosity, and pot water-holding capacity.

3. Dose lookup
   One row per soil and phosphorus level, including relative P level, absolute P dose, and fertilizer source quantities.

## Main clean datasets planned

```text
experiment_results_clean.csv
soil_characterization_replicates_clean.csv
dose_lookup_clean.csv
```

Derived datasets to be generated later by scripts:

```text
soil_characterization_summary.csv
experiment_analysis_dataset.csv
extractor_long_format.csv
```

## Main analytical targets

The main analytical questions are:

1. Which extractor best predicts plant phosphorus uptake?
2. Which extractor best predicts dry matter response?
3. How do extractor recovery rates vary among soils?
4. How do soil attributes, especially P-rem, pH, calcium, texture, and organic carbon, affect extractor performance?
5. Can critical phosphorus levels be recalculated for each extractor using relative plant response?
6. Are Mehlich-3 and Olsen more reliable than Mehlich-1 in alkaline or calcium-rich semiarid soils?

## Main extractors

The project evaluates:

```text
Mehlich-1
Mehlich-3
Anion exchange resin
Olsen
```

## Main plant responses

The main plant response variables are expected to include:

```text
dry_matter_cut1
dry_matter_cut2
dry_matter_total
plant_P_cut1
plant_P_cut2
P_uptake_cut1
P_uptake_cut2
P_uptake_total
relative_dry_matter
relative_P_uptake
```

## Data privacy

Original experimental files, Excel workbooks, SAS outputs, and real processed datasets are private and must not be committed to GitHub.

Public data should be synthetic, anonymized, or explicitly approved for release.

## Reproducibility principle

Old SAS outputs may be retained as historical reference, but the current project should reconstruct the analysis from cleaned raw data using Python scripts.

The analysis should not depend on old manual spreadsheet formulas or SAS output files.
