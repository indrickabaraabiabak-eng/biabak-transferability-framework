# Changelog
## 1.0.1 (2026-09-29)
- Public-release sanitation: removed record-level borehole records, locality names, coordinates and row-level hydraulic/covariate outputs from redistributed supplementary material because the provider conditions do not permit redistribution.
- Preserved supplementary table numbering by replacing restricted record-level tables with data-access statements.
- Public figures omit individual borehole locations.
- Added NNDM phi sensitivity, drainage-density sensitivity summaries and the final Step 3C validation-geometry figure.
- Fixed `biabak run --skip-benchmark` so benchmark-dependent downstream steps are skipped consistently.

## 1.0.0 (2026-09-24)
- First release accompanying the submitted article: archive audit, structure diagnostics, validation
  designs with NNDM, fold-baseline skill with excess-risk AUC, area of applicability, synthetic benchmark
  (30 replicates per scenario, extended to 330 replicates for the null scenario and 130 per covariate share, plus 40 replicates per cell for range sensitivity), unit tests.
