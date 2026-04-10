import math

PROJECT = "skytruth-tech"

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

# Natural Forests of the World probability threshold
NFW_THRESHOLD = 0.5

# Parameters for reduceRegion raster calculations
CRS = "EPSG:3857"
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

# Earth Engine asset IDs
PAS_ASSET_ID = "WCMC/WDPA/current/polygons"
OECMS_ASSET_ID = "WCMC/WDOECM/current/polygons"
GLC_ASSET_ID = "projects/sat-io/open-datasets/GLC-FCS30D/annual"
HGFC_ASSET_ID = "UMD/hansen/global_forest_change_2024_v1_12"
GPW_ASSET_ID = "projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c"
NFW_ASSET_ID = (
    "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
)
COUNTRIES_ASSET_ID = "USDOS/LSIB_SIMPLE/2017"

# Repository Based Filepaths
REPO_DATA_DIR = "data/"

# Variables for psm_grid_creation.py
WDPA_TEST_SITE_GEOJSON = REPO_DATA_DIR + "test_sites.geojson"
WDPA_TEST_SITE_10M_BUFFER = REPO_DATA_DIR + "test_sites_10km_4087.parquet"
WDPA_TEST_SITE_50M_BUFFER = REPO_DATA_DIR + "test_sites_50km_4087.parquet"
WDPA_EXCLUSION_ZONE = REPO_DATA_DIR + "test_sites_exclusion_zone_4087.parquet"
WDPA_WIDER_LANDSCAPE = REPO_DATA_DIR + "test_sites_wider_landscape_4087.parquet"
WDPA_1KM_GRID = REPO_DATA_DIR + "test_sites_1km_grid_4087.parquet"
WDPA_1KM_PSM_GRID = REPO_DATA_DIR + "test_sites_TPA_PSM_GRID.parquet"

# Variables for global_grid_creation.py
PSM_TEST_AOI = REPO_DATA_DIR + "Ghana.geojson"
PSM_TEST_PAS = REPO_DATA_DIR + "Ghana_PAs.geojson"
PSM_TEST_CELLS = REPO_DATA_DIR + "Ghana_PSM_CELLS.parquet"

# Variables for sample_point_grid.py
PSM_MIN_DISTANCE = 3000
TREATMENT_N = 5000
CONTROL_N = 10000
RAND_SEED = 42
OVERSAMPLE_FACTOR = 5.0
MAX_ROUNDS = 12

# General PSM parameters
PSM_CRS = 6933
PSM_CELL_SIZE = 1000
PSM_CONTROL_BUFFER = 10000

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
