@echo off
chcp 65001 >nul
set "BIABAK_BOREHOLE_XLSX=C:\path\to\DATA_ABARA_224_forages.xlsx"
set "BIABAK_COVARIATE_XLSX=C:\path\to\Covariables_forages_extraites.xlsx"
set "BIABAK_RASTER_DIR=C:\path\to\rasters"
set "BIABAK_GEOLOGY_DIR=C:\path\to\geology"
set "BIABAK_SOIL_SHP=C:\path\to\soil_region_de_centre.shp"
set "BIABAK_OUT_DIR=%~dp0outputs"
