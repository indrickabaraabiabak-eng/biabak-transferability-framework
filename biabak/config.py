"""Central configuration for the transferability-audit workflow.

Every constant used downstream is declared here so that a reader can rerun the
analysis and obtain identical numbers. Nothing in this file is estimated from
the data.
"""
from pathlib import Path
import os

# ----------------------------------------------------------------------------
# Input/output paths
# Personal machine paths are supplied through environment variables.
# Portable defaults are relative to the project directory.
# ----------------------------------------------------------------------------
def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _env_path("BIABAK_DATA_DIR", PROJECT_ROOT / "data")
BOREHOLE_XLSX = _env_path("BIABAK_BOREHOLE_XLSX", DATA_DIR / "DATA_ABARA_224_forages.xlsx")
BOREHOLE_SHEET = "Donnees_224_forages"
COVARIATE_XLSX = _env_path("BIABAK_COVARIATE_XLSX", DATA_DIR / "Covariables_forages_extraites.xlsx")
RASTER_DIR = _env_path("BIABAK_RASTER_DIR", DATA_DIR / "rasters")
GEOLOGY_DIR = _env_path("BIABAK_GEOLOGY_DIR", DATA_DIR / "geology")
SOIL_SHP = _env_path("BIABAK_SOIL_SHP", DATA_DIR / "soil_region_de_centre.shp")
OUT_DIR = _env_path("BIABAK_OUT_DIR", PROJECT_ROOT / "outputs")

# ----------------------------------------------------------------------------
# Coordinate reference systems
# ----------------------------------------------------------------------------
CRS_GEOGRAPHIC = 4326      # borehole coordinates as delivered
CRS_ANALYSIS = 32632       # UTM zone 32N, used for every distance in the study

# ----------------------------------------------------------------------------
# Raster covariates: column name -> (file stem, type)
# Drainage.tif is deliberately absent: the extraction log states that its
# physical meaning could not be established.
# ----------------------------------------------------------------------------
RASTER_COVARIATES = {
    "elevation_m": ("DEM_100m", "continuous"),
    "slope_deg": ("Slope_100m", "continuous"),
    "profile_curvature": ("profil_curve", "continuous"),
    "plan_curvature": ("plan_curve", "continuous"),
    "lineament_density": ("Densite_lineique", "continuous"),
    "rainfall_mm_yr": ("Pluvio_mm_an", "continuous"),
    "aet_mm_yr": ("ETR_mm_an", "continuous"),
    "ndvi": ("NDVI_bon", "continuous"),
    "ndbi": ("NDBI_bon", "continuous"),
    "lithology_code": ("Geol_raster", "categorical"),
    "landcover_code": ("Landuse_bon", "categorical"),
}
FALLBACK_RADIUS_PIXELS = 3   # documented fallback when the borehole pixel is empty

# ----------------------------------------------------------------------------
# Analytical populations
# ----------------------------------------------------------------------------
# Five productive boreholes outside the common raster support (sheet Forages_219)
EXCLUDED_FROM_COMMON_SUPPORT = [1, 2, 37, 121, 125]   # values of column "N°"

# ----------------------------------------------------------------------------
# Variography
# ----------------------------------------------------------------------------
LAG_KM = 2.0
HMAX_KM = 70.0
MIN_PAIRS = 30
SHORT_BINS_KM = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
DIRECTIONS_DEG = {"N-S": 0.0, "NE-SW": 45.0, "E-W": 90.0, "SE-NW": 135.0}
ANGULAR_TOL_DEG = 22.5
N_PERMUTATIONS = 999
N_PERMUTATIONS_DIRECTIONAL = 499
LAG_SENSITIVITY_KM = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
DECLUSTER_CELLS_KM = [5.0, 10.0, 15.0, 20.0, 30.0]
DECLUSTER_N_ORIGINS = 25
RANGE_BOUNDS_KM = (0.5, 150.0)

# Dense cluster around the capital, used only for the cluster decomposition
CLUSTER_CENTRE_LATLON = (3.866, 11.516)
CLUSTER_RADIUS_KM = 15.0
LOCAL_WINDOW_KM = 10.0
LOCAL_MIN_POINTS = 5

# Pairs closer than this are listed as possible duplicate records
DUPLICATE_CHECK_M = 20.0

SEED = 20260924
FAULT_SHP = GEOLOGY_DIR / "Faille.shp"
