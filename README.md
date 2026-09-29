# BIABAK Transferability Framework

The BIABAK Transferability Framework evaluates whether a model that predicts a point-scale quantity (here borehole productivity) from
spatial covariates keeps its skill at the distances where it is meant to be used, and it reports which
effect sizes a null result can and cannot exclude.

## What it does
1. Archive audit (populations, digit preference, close pairs, duplicate log).
2. Spatial-structure diagnostics before interpolation: permutation envelopes, competing variogram
   families (spherical, exponential, cardinal sine, J-Bessel), identifiability, cluster decomposition.
3. Leakage-controlled geostatistical and machine-learning predictors (all preprocessing and variogram
   fitting inside each training fold).
4. Validation designs (random, repeated random, k-means blocks, square blocks, NNDM leave-one-out)
   with the distance from each test point to the nearest training point, compared with the prediction domain.
5. Skill against fold-specific baselines (R2cv, Brier skill score, excess-risk AUC) with bootstrap
   intervals and Holm adjustment.
6. Area of applicability (dissimilarity index).
7. Synthetic detection benchmark on the real network, with binomial intervals and a one-sided
   compatibility rule.

## Installation
BIABAK requires Python 3.12.

Clone the repository, enter the project directory, create a virtual environment, and install the framework in editable mode:

    git clone https://github.com/indrickabaraabiabak-eng/biabak-transferability-framework.git
    cd biabak-transferability-framework
    py -3.12 -m venv .venv
    .venv\Scripts\activate
    python -m pip install -e .

After installation, verify that the command-line interface is available:

    biabak --help

## Running
On Windows, copy `local_env.example.bat` to `local_env.bat`, edit the paths in the local copy, then run:

    call local_env.bat

The local file is excluded from version control.

The following environment variables can be used to point BIABAK to the required data and output locations:

- `BIABAK_DATA_DIR`
- `BIABAK_BOREHOLE_XLSX`
- `BIABAK_COVARIATE_XLSX`
- `BIABAK_RASTER_DIR`
- `BIABAK_GEOLOGY_DIR`
- `BIABAK_SOIL_SHP`
- `BIABAK_OUT_DIR`

Run the complete workflow with:

    biabak run

To skip the computationally intensive synthetic benchmark:

    biabak run --skip-benchmark

To install the optional test dependencies and run the unit tests:

    python -m pip install -e .[test]
    python -m pytest -q tests

All analytical constants (seeds, lags, permutations, bootstrap resamples, NNDM settings and benchmark design) are declared in `biabak/config.py` or explicitly in the corresponding analysis script.

## Inputs and data-access restrictions
The complete workflow requires the borehole inventory and the covariate raster/vector layers described in `biabak/config.py`. The original borehole archive used in the article cannot be redistributed under the provider conditions. Individual borehole records, locality names, coordinates, hydraulic measurements and row-level derived covariates are therefore not included in the public release. Public supplementary workbooks preserve the cited table numbering, but record-level slots (Tables S3, S4, S7, S8 and S37) contain data-access statements rather than individual records. Aggregated diagnostics, model-performance summaries, benchmark results and other non-record-level derived outputs are provided in the remaining supplementary tables. Public figures omit individual borehole locations. Authorized users who have lawful access to the original archive can regenerate the full internal record-level outputs locally with the released code.

## Citation
Zenodo DOI: `10.5281/zenodo.23038162``

See `CITATION.cff` for the complete citation metadata.
