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
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Running
The workflow uses portable relative paths by default. Personal paths must not be written into the source code. Input locations can instead be supplied through environment variables defined before execution.

On Windows, copy `local_env.example.bat` to `local_env.bat`, edit the paths in the local copy, then run `call local_env.bat` before starting the workflow. The local file is excluded from version control.

Supported variables are:

- `BIABAK_DATA_DIR`
- `BIABAK_BOREHOLE_XLSX`
- `BIABAK_COVARIATE_XLSX`
- `BIABAK_RASTER_DIR`
- `BIABAK_GEOLOGY_DIR`
- `BIABAK_SOIL_SHP`
- `BIABAK_OUT_DIR`

If no variables are supplied, inputs are sought under the project `data/` directory and outputs are written to `outputs/`.

    python run_all.py                    # complete workflow
    python run_all.py --skip-benchmark   # workflow without the synthetic benchmark
    python -m pytest -q tests            # unit tests

All analytical constants (seeds, lags, permutations, bootstrap resamples, NNDM settings and benchmark design) are declared in `biabak/config.py` or explicitly in the corresponding analysis script.

## Inputs
The complete workflow requires the borehole inventory and the covariate raster/vector layers described in `biabak/config.py`. The original borehole archive used in the article cannot be redistributed under the provider conditions. Redistributable derived results are provided as Supplementary Tables S1 to S42 in three workbooks: S1-S23 for archive and spatial-structure diagnostics, S24-S34 for validation and predictive performance, and S35-S42 for applicability and the synthetic detection benchmark.

## Citation
See `CITATION.cff`.
