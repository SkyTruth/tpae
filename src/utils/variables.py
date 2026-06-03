import math

PROJECT = "skytruth-tech"
GCS_BUCKET = "tpae"

# Earth Engine asset IDs
PAS_ASSET_ID = "WCMC/WDPA/current/polygons"
OECMS_ASSET_ID = "WCMC/WDOECM/current/polygons"
GLC_ASSET_ID = "projects/sat-io/open-datasets/GLC-FCS30D/annual"
HGFC_ASSET_ID = "UMD/hansen/global_forest_change_2025_v1_13"
GPW_ASSET_ID = "projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c"
NFW_ASSET_ID = (
    "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
)
COUNTRIES_ASSET_ID = "USDOS/LSIB_SIMPLE/2017"
BIOME_ASSET_ID = "RESOLVE/ECOREGIONS/2017"
GLO30_ASSET_ID = "COPERNICUS/DEM/GLO30"
ATC_ASSET_ID = "projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0"
POP_ASSET_ID = "JRC/GHSL/P2023A/GHS_POP/2000"

TPAE_ASSET_FOLDER = "projects/skytruth-tech/assets/TPAE/"
GAEZ_WHEAT_ASSET_ID = TPAE_ASSET_FOLDER + "GAEZ_wheat"
GAEZ_RICE_ASSET_ID = TPAE_ASSET_FOLDER + "GAEZ_dryland_rice"
GAEZ_MAIZE_ASSET_ID = TPAE_ASSET_FOLDER + "GAEZ_maize"
GAEZ_SOYBEAN_ASSET_ID = TPAE_ASSET_FOLDER + "GAEZ_soybean"
HUMAN_FOOTPRINT_ASSET_ID = TPAE_ASSET_FOLDER + "human_footprint_1993"

# WDPAID numbers of selected test PAs and OECMs
TEST_SITE_IDS = [
    555714961,
    1543,
    555557937,
    93538,
    2008,
    916,
    303317,
    352159,
    555626124,
    7949,
    214,
    306522,
    555786096,
    555599263,
    555759266,
    9436,
    1250,
    68399,
    11116292,
    67967,
    555512003,
    555752316,
    166970,
    26654,
    101664,
    555766202,
    2017,
    10711,
    164,
    555784006,
]

# Habitat loss analysis period
ANALYSIS_START_YR = 2018
ANALYSIS_END_YR = 2022

# Global Land Cover original class values
GLC_CLASSES = [
    10,
    11,
    12,
    20,
    51,
    52,
    61,
    62,
    71,
    72,
    81,
    82,
    91,
    92,
    120,
    121,
    122,
    130,
    140,
    150,
    152,
    153,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    190,
    200,
    201,
    202,
    210,
    220,
    0,
]

# Global Land Cover palette
GLC_PALETTE = [
    "#ffff64",
    "#ffff64",
    "#ffff00",
    "#aaf0f0",
    "#4c7300",
    "#006400",
    "#a8c800",
    "#00a000",
    "#005000",
    "#003c00",
    "#286400",
    "#285000",
    "#a0b432",
    "#788200",
    "#966400",
    "#964b00",
    "#966400",
    "#ffb432",
    "#ffdcd2",
    "#ffebaf",
    "#ffd278",
    "#ffebaf",
    "#00a884",
    "#73ffdf",
    "#9ebb3b",
    "#828282",
    "#f57ab6",
    "#66cdab",
    "#444f89",
    "#c31400",
    "#fff5d7",
    "#dcdcdc",
    "#fff5d7",
    "#0046c8",
    "#ffffff",
    "#ffffff",
]

# Global Land Cover class labels
GLC_LABELS = {
    1: "Rainfed cropland",
    2: "Herbaceous cover cropland",
    3: "Tree or shrub cover (Orchard) cropland",
    4: "Irrigated cropland",
    5: "Open evergreen broadleaved forest",
    6: "Closed evergreen broadleaved forest",
    7: "Open deciduous broadleaved forest (0.15<fc<0.4)",
    8: "Closed deciduous broadleaved forest (fc>0.4)",
    9: "Open evergreen needle-leaved forest (0.15< fc <0.4)",
    10: "Closed evergreen needle-leaved forest (fc >0.4)",
    11: "Open deciduous needle-leaved forest (0.15< fc <0.4)",
    12: "Closed deciduous needle-leaved forest (fc >0.4)",
    13: "Open mixed leaf forest (broadleaved and needle-leaved)",
    14: "Closed mixed leaf forest (broadleaved and needle-leaved)",
    15: "Shrubland",
    16: "Evergreen shrubland",
    17: "Deciduous shrubland",
    18: "Grassland",
    19: "Lichens and mosses",
    20: "Sparse vegetation (fc<0.15)",
    21: "Sparse shrubland (fc<0.15)",
    22: "Sparse herbaceous (fc<0.15)",
    23: "Swamp",
    24: "Marsh",
    25: "Flooded flat",
    26: "Saline",
    27: "Mangrove",
    28: "Salt marsh",
    29: "Tidal flat",
    30: "Impervious surfaces",
    31: "Bare areas",
    32: "Consolidated bare areas",
    33: "Unconsolidated bare areas",
    34: "Water body",
    35: "Permanent ice and snow",
    36: "Filled value",
}

# Driver of habitat loss class labels
DRIVER_LABELS = {
    1: "Cropland",
    2: "Built-up Land",
    3: "Pasture",
    4: "Deforestation without conversion",
}

BIOME_PALETTE = [
    "#38A700",
    "#CCCD65",
    "#88CE66",
    "#00734C",
    "#458970",
    "#7AB6F5",
    "#FEAA01",
    "#FEFF73",
    "#BEE7FF",
    "#D6C39D",
    "#FFEAAF",
    "#FE0000",
    "#CC6767",
    "#FE01C4",
]

BIOME_LABELS = {
    1: "Tropical & Subtropical Moist Broadleaf Forests",
    2: "Tropical & Subtropical Dry Broadleaf Forests",
    3: "Tropical & Subtropical Coniferous Forests",
    4: "Temperate Broadleaf & Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests/Taiga",
    7: "Tropical & Subtropical Grasslands, Savannas & Shrublands",
    8: "Temperate Grasslands, Savannas & Shrublands",
    9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    11: "N/A",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves",
}

# Natural Forests of the World probability threshold
NFW_THRESHOLD = 0.5

# WKT for EPSG:6933 (NSIDC EASE-Grid 2.0 Global projection)
# Earth Engine is unable to parse "EPSG:6933" directly
# Equal-area projection used for calculations in meters
WKT_6933 = """
    PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Cylindrical_Equal_Area"],
        PARAMETER["standard_parallel_1",30],
        PARAMETER["central_meridian",0],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","6933"]]
    """

# Parameters for reduceRegion raster calculations
EE_CRS_METERS = WKT_6933
SCALE = 30
MAX_PIXELS = 1e13

# Parameters for habitat intactness calculations
INTERACTION_DISTANCE = 500  # meters
BETA = 1 / INTERACTION_DISTANCE  # controls the rate of exponential decay
KERNEL_RADIUS_METERS = (
    5 * INTERACTION_DISTANCE
)  # should be proportional to beta to truncate the tail and reduce unnessary computation expense
INTACTNESS_SCALE = 60  # pixel size in meters
KERNEL_RADIUS_PIXELS = math.ceil(KERNEL_RADIUS_METERS / INTACTNESS_SCALE)
KERNEL_SIZE = KERNEL_RADIUS_PIXELS * 2 + 1  # width and height of the kernel
TILE_SCALE = 4

# Parameters for habitat loss calculations
OPENING_RADIUS_LOSS = 30  # meters

# General PSM parameters
PSM_CELL_SIZE = 1000
GPD_CRS_PARQUET = "EPSG:4326"
GPD_CRS_METERS = "EPSG:6933"
RAND_SEED = 42
CALIPER = 0.2
N_NEIGHBORS = 4 # number of control cells for every treatment cell

# Parameters for global propensity score model
COVARIATES = ["elevation", "slope", "treecover2000", "travel_time", "log_pop_density", "human_footprint", "ag_suitability"]
TOTAL_POINTS = 100000 # total number of samples to collect globally
TREAT_CONTROL = (1, 2) # ratio of protected to unprotected samples
MIN_PER_STRATUM = 50 # minimum number of samples per stratum
PSM_SAMPLES_PREFIX = "psm_samples/covariates_7/" # prefix for PSM samples in GCS bucket
STRATA_ASSET_ID = f"projects/{PROJECT}/assets/TPAE/strata_1km"

# Parameters for treatment cell sampling
PA_AREA_THRESHOLD = 500000000 # 500 km2
SAMPLE_AREA_PCT = 0.03 # sample this percentage of the PA's area

# Parameters for control cell sampling
CONTROL_INNER_BUFFER = 10_000 # minimum distance (m) from PA
CONTROL_OUTER_BUFFER = 200_000 # maximum distance (m) from PA
CONTROL_SPACING = 3000 # minimum distance (m) between control cells
CONTROL_N_SAMPLES = 2000 # number of control cells to sample for each PA

#------------------------------------------------------------------------------------------------
# FILE PATHS
#------------------------------------------------------------------------------------------------

# Data directory
REPO_DATA_DIR = "data/"

# Inputs:
#----------
# Geojson of 30 test PAs
TEST_SITES_GEOJSON = REPO_DATA_DIR + "test_sites_cleaned.geojson"

# Outputs:
#----------
# global_psm.ipynb
STATE_FILE = REPO_DATA_DIR + "psm/psm_tile_state.json"
# get_treatment_cells.py
TREATMENT_CELLS = REPO_DATA_DIR + "treatment_cells.parquet"
# get_control_cells.ipynb
CONTROL_CELLS = REPO_DATA_DIR + "control_cells.parquet"

# Archived:
#----------
# global_grid_creation.py
PSM_GLOBAL_GRID = REPO_DATA_DIR + "Ghana_global_grid.parquet"
PSM_TEST_AOI = REPO_DATA_DIR + "Ghana.geojson"
PSM_TEST_PAS = REPO_DATA_DIR + "Ghana_PAs.geojson"
# psm_grid_creation.py
BUFFER_10KM = REPO_DATA_DIR + "test_sites_10km_4326.parquet"
BUFFER_50KM = REPO_DATA_DIR + "test_sites_50km_4326.parquet"
EXCLUSION_ZONE = REPO_DATA_DIR + "test_sites_exclusion_zone_4326.parquet"
WIDER_LANDSCAPE = REPO_DATA_DIR + "test_sites_wider_landscape_4326.parquet"
GRID_1KM = REPO_DATA_DIR + "test_sites_1km_grid_4326.parquet"
PSM_GRID_1KM = REPO_DATA_DIR + "test_sites_TPA_PSM_GRID.parquet"