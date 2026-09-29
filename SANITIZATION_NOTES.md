# BIABAK public-release sanitation notes

Provider conditions do not permit redistribution of individual borehole records or row-level derived data.

## Replace in the repository before a new public release
- `README.md`
- `CHANGELOG.md`
- `biabak/config.py`
- `export_step1.py`
- `export_step3.py`
- `figures_manuscript.py`
- `figures_step1.py`
- `figures_step3.py`
- `run_step2c_drainage_sensitivity.py`
- `outputs/step1/Supplementary_Tables_Step1_Variography.xlsx`
- `outputs/step3/Supplementary_Tables_Step3_Applicability_Benchmark.xlsx`

The sanitized Step-1 workbook preserves table numbering but replaces Tables S3, S4, S7 and S8 with data-access statements. The sanitized Step-3 workbook replaces Table S37 with a data-access statement.

## Safe sensitivity outputs included
- Aggregate drainage sensitivity summaries (`summary`, `deltas`, `metadata`) and a sanitized workbook.
- Aggregate phi-sensitivity summary CSV/XLSX only.
- Final Step-3C S5 figure and its generation script.

## Do not publish / do not add to Git
- `outputs/step2/drainage_sensitivity/drainage_values_224.csv`
- any drainage workbook containing the `drainage_values_224` record-level sheet
- `outputs/step2/phi_sensitivity/checkpoints/`
- `outputs/step2/phi_sensitivity/nndm_info_*.pkl`
- any workbook or table containing locality names, exact coordinates, row-level hydraulic values, or row-level derived covariates

## Figures that must be regenerated before public use
The patched scripts suppress exact borehole locations, but the existing image files in the old release still contain them. Regenerate and replace:
- `outputs/manuscript_figures/Figure_2.png/.pdf`
- `outputs/step1/Fig_S4_cluster_shortrange_proportional.png/.pdf`
- `outputs/step3/Fig_B1_area_of_applicability.png/.pdf`

The patched Figure 2 omits individual borehole points. The patched Step-1 cluster figure replaces the exact-coordinate panel with aggregate inside/outside counts. The patched AoA figure omits point overlays.

## Important history issue
A new commit or v1.0.1 release does not erase restricted files already present in earlier Git commits/tags or an existing archived release. Treat the old public version/history as a separate remediation task before considering the disclosure fully resolved.
