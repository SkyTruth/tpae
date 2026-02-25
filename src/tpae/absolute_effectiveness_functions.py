import ee
from math import sqrt
from tpae.utils.variables import (
    PROJECT,
    TEST_SITE_IDS,
    ANALYSIS_START_YR,
    ANALYSIS_END_YR,
    GLC_CLASSES,
    NFW_THRESHOLD,
    MAX_EDGE_DIST,
    OPENING_RADIUS_EDGE,
    MAX_PATCH_SIZE,
    OPENING_RADIUS_LOSS,
    CRS,
    SCALE,
    MAX_PIXELS,
)

ee.Authenticate()
ee.Initialize(project=PROJECT)

# Data imports
PAS = ee.FeatureCollection("WCMC/WDPA/current/polygons")
OECMS = ee.FeatureCollection("WCMC/WDOECM/current/polygons")
GLC = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
HGFC = ee.Image("UMD/hansen/global_forest_change_2024_v1_12")
GPW = ee.ImageCollection("projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c")
NFW = ee.ImageCollection(
    "projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection"
)


def get_test_sites():
    """Get a set of selected PAs for testing.

    Returns:
        ee.FeatureCollection of selected test sites
    """
    all_sites = (
        ee.FeatureCollection([PAS, OECMS])
        .flatten()
        .filter(ee.Filter.eq("REALM", "Terrestrial"))
    )
    test_sites = all_sites.filter(ee.Filter.inList("SITE_ID", TEST_SITE_IDS))
    return test_sites


def set_start_yr(test_sites, site_id):
    """Define analysis start year. (If a PA was designated after the start of the analysis period, use the designation year as the start year.)

    Args:
        test_sites: ee.FeatureCollection of selected PAs
        site_id: Integer ID of site to analyze

    Returns:
        Integer year to start analysis
    """
    designation_yr = (
        test_sites.filter(ee.Filter.eq("SITE_ID", site_id))
        .first()
        .get("STATUS_YR")
        .getInfo()
    )
    return max(ANALYSIS_START_YR, designation_yr)


def check_start_yr(start_yr):
    """Check if start year is valid. (PA must have been designated at least one year before the end of the analysis period.)

    Args:
        start_yr: Integer year to start analysis
    """
    if start_yr > (ANALYSIS_END_YR - 1):
        print(f"PA was designated too recently ({start_yr}) to analyze effectiveness.")
    else:
        print(f"Start year ({start_yr}) is valid.")


def get_site_geom(test_sites, site_id):
    """Get geometry of a specific PA.

    Args:
        test_sites: ee.FeatureCollection of selected PAs
        site_id: Integer ID of site to analyze

    Returns:
        ee.Geometry of selected site
    """
    return test_sites.filter(ee.Filter.eq("SITE_ID", site_id)).geometry()


def process_GLC(test_sites, start_yr):
    """Process Global Land Cover Change data.

    Args:
        test_sites: ee.FeatureCollection of selected PAs
        start_yr: Integer year to start analysis

    Returns:
        ee.Image with remapped GLC bands for each year in analysis period
    """
    # Mosaic images in the collection that intersect test sites
    GLC_mosaic = GLC.filterBounds(test_sites).mosaic()
    # Select and rename bands corresponding to analysis period
    analysis_years = list(range(start_yr, ANALYSIS_END_YR + 1))
    band_names = [f"b{year - 2000 + 1}" for year in analysis_years]
    new_band_names = [f"GLC_{year}" for year in analysis_years]
    GLC_selected = GLC_mosaic.select(band_names, new_band_names)

    # Remap class values to 1-36
    def remap_classes(band):
        return (
            GLC_selected.select(band)
            .remap(GLC_CLASSES, ee.List.sequence(1, len(GLC_CLASSES)), defaultValue=0)
            .rename([band])
        )

    remapped_bands = [remap_classes(band) for band in new_band_names]
    GLC_remapped = ee.Image.cat(remapped_bands)
    return GLC_remapped


def process_GPW(start_yr):
    """Process Global Pasture Watch data.

    Args:
        start_yr: Integer year to start analysis

    Returns:
        ee.Image with GPW bands for each year in analysis period
    """
    year_strings = [str(year) for year in range(start_yr, ANALYSIS_END_YR + 1)]
    GPW_filtered = GPW.filter(ee.Filter.inList("system:index", year_strings)).toBands()
    GPW_renamed = GPW_filtered.rename([f"GPW_{year}" for year in year_strings])
    return GPW_renamed.unmask()


def process_NFW(test_sites):
    """Process Natural Forests of the World (2020) data.

    Args:
        test_sites: ee.FeatureCollection of selected PAs

    Returns:
        ee.Image with thresholded natural forest binary
    """
    # Mosaic images in the collection that intersect test sites
    NFW_mosaic = NFW.filterBounds(test_sites).mosaic()
    # Set probability threshold
    NFW_thresholded = NFW_mosaic.gte(NFW_THRESHOLD)
    return NFW_thresholded


def process_HGFC(start_yr):
    """Process Hansen Global Forest Change data.

    Args:
        start_yr: Integer year to start analysis

    Returns:
        ee.Image with forest loss year data masked to analysis period
    """
    # Select forest loss band
    HGFC_selected = HGFC.select("lossyear")
    # Mask to forest loss within analysis period
    analysis_mask = HGFC_selected.gte(start_yr - 2000).And(
        HGFC_selected.lte(ANALYSIS_END_YR - 2000)
    )
    HGFC_masked = HGFC_selected.updateMask(analysis_mask)
    return HGFC_masked.unmask()


def get_habitat_raster(GLC_processed, HGFC_processed, GPW_processed, NFW_processed):
    """Create habitat extent raster for ANALYSIS_END_YR by masking out non-habitat areas.

    Args:
        GLC_processed: ee.Image with processed Global Land Cover data
        HGFC_processed: ee.Image with processed Hansen Global Forest Change data
        GPW_processed: ee.Image with processed Global Pasture Watch data
        NFW_processed: ee.Image with processed Natural Forests of the World data

    Returns:
        ee.Image with classified habitat extent for ANALYSIS_END_YR
    """
    # Select GLC data for ANALYSIS_END_YR
    GLC_current = GLC_processed.select(f"GLC_{ANALYSIS_END_YR}")
    # Select cropland (1-4) and impervious surfaces (30)
    anthro_classes = ee.List([1, 2, 3, 4, 30])
    anthro_mask = GLC_current.remap(
        anthro_classes, ee.List.repeat(1, anthro_classes.size()), defaultValue=0
    )
    # Select any forest loss pixels during the analysis period
    forest_loss_mask = HGFC_processed.gt(0)
    # Select any cultivated grassland pixels using GPW
    GPW_current = GPW_processed.select(f"GPW_{ANALYSIS_END_YR}")
    pasture_mask = GPW_current.eq(1)
    # Select any pixels of "Forest" class that do not coincide with Natural Forests
    closed_forest_classes = ee.List([6, 8, 10, 12, 14])
    planted_forest_mask = (
        GLC_current.remap(
            closed_forest_classes,
            ee.List.repeat(1, closed_forest_classes.size()),
            defaultValue=0,
        ).updateMask(NFW_processed.eq(0))
    ).unmask(0)
    # Preserve GPW grassland class where it overlaps anthro or planted-forest areas.
    grassland_mask = GPW_current.eq(2)
    grassland_override = grassland_mask.And(anthro_mask.add(planted_forest_mask).gt(0))
    anthro_mask = anthro_mask.where(grassland_override, 0)
    planted_forest_mask = planted_forest_mask.where(grassland_override, 0)

    # Combine all non-habitat masks and invert; apply mask to landcover raster
    non_habitat_mask = (
        anthro_mask.add(forest_loss_mask).add(pasture_mask).add(planted_forest_mask)
    )
    habitat_mask = non_habitat_mask.eq(0).Or(grassland_override)
    habitat_raster = GLC_current.where(grassland_override, 18).updateMask(habitat_mask)
    return habitat_raster


def calc_habitat_extent_score(habitat_raster, site_geom):
    """Calculate Habitat Extent score for ANALYSIS_END_YR within a PA.

    Args:
        habitat_raster: ee.Image with habitat extent
        site_geom: ee.Geometry of the PA

    Returns:
        Float proportion of site area that is habitat (0-1)
    """
    # Calculate total area of site
    site_area = site_geom.area().getInfo()
    # Calculate total area of habitat within site
    habitat_area = (
        ee.Image.pixelArea()
        .updateMask(habitat_raster)
        .reduceRegion(
            ee.Reducer.sum(), site_geom, scale=SCALE, crs=CRS, maxPixels=MAX_PIXELS
        )
        .get("area")
        .getInfo()
    )
    habitat_proportion = min(habitat_area / site_area, 1)
    return habitat_proportion


def get_edge_distance_raster(habitat_raster, site_geom):
    """Calculate distance from each habitat pixel to nearest non-habitat pixel.

    Args:
        habitat_raster: ee.Image with habitat extent
        site_geom: ee.Geometry of the PA

    Returns:
        ee.Image with edge distance values for habitat pixels
    """
    # Buffer site to max edge distance
    site_buffered = site_geom.buffer(MAX_EDGE_DIST + OPENING_RADIUS_EDGE)
    # Reproject
    projection = ee.Projection("EPSG:3857").atScale(30)
    habitat_reprojected = habitat_raster.clip(site_buffered).reproject(projection)
    # Create and invert habitat binary (habitat = 0, non-habitat = 1)
    habitat_binary = habitat_reprojected.unmask().eq(0)
    # Morphologically open (erode and dilate) the habitat binary to remove tiny areas of non-habitat
    habitat_binary_opened = habitat_binary.focalMin(
        radius=OPENING_RADIUS_EDGE, kernelType="square", units="meters"
    ).focalMax(radius=OPENING_RADIUS_EDGE, kernelType="square", units="meters")
    # Calculate distance to nearest non-habitat pixel
    edge_distance = habitat_binary_opened.distance(
        ee.Kernel.euclidean(MAX_EDGE_DIST, units="meters")
    )
    # Clean edge distance layer
    edge_distance_cleaned = (
        edge_distance.min(MAX_EDGE_DIST)  # Cap values to saturate at max edge distance
        .unmask(MAX_EDGE_DIST)  # Set masked values (values beyond the kernel) to max
        .clip(site_geom)  # Clip to PA
        .selfMask()  # Mask all 0 (non-habitat) pixels
        .rename("edge_distance")
    )  # Rename band for clarity
    return edge_distance_cleaned


def calc_edge_distance_score(edge_distance_raster, site_geom):
    """Calculate a PA's Edge Distance score from edge distance raster.

    Args:
        edge_distance_raster: ee.Image with edge distance values for habitat pixels
        site_geom: ee.Geometry of the PA

    Returns:
        Float average edge distance, scaled to 0-1
    """
    avg_edge_distance = (
        edge_distance_raster.reduceRegion(
            ee.Reducer.mean(), site_geom, scale=SCALE, crs=CRS, maxPixels=MAX_PIXELS
        )
        .get("edge_distance")
        .getInfo()
    )
    avg_edge_distance_scaled = avg_edge_distance / MAX_EDGE_DIST
    return avg_edge_distance_scaled


def get_patch_size_raster(habitat_raster, site_geom):
    """Calculate size of connected habitat patches.

    Args:
        habitat_raster: ee.Image with habitat extent
        site_geom: ee.Geometry of the PA

    Returns:
        ee.Image with patch size values for each habitat pixel
    """
    patch_size = (
        habitat_raster.gt(0)  # Make habitat binary
        .connectedPixelCount(maxSize=1024, eightConnected=False)
        .rename("patch_size")
        .clip(site_geom)
    )
    return patch_size


def calc_patch_size_score(patch_size_raster, site_geom):
    """Calculate a PA's Patch Size score from patch size raster.

    Args:
        patch_size_raster: ee.Image with patch size values for habitat pixels
        site_geom: ee.Geometry of the PA

    Returns:
        Float average patch size, scaled to 0-1
    """
    # Calculate scale for patch size reduction
    patch_scale = sqrt((MAX_PATCH_SIZE * 1000000) / 1024)

    avg_patch_size = (
        patch_size_raster.reduceRegion(
            ee.Reducer.mean(),
            site_geom,
            scale=patch_scale,
            crs=CRS,
            maxPixels=MAX_PIXELS,
        )
        .get("patch_size")
        .getInfo()
    )
    avg_patch_size_scaled = avg_patch_size / 1024
    return avg_patch_size_scaled


def calc_intactness_score(edge_distance_score, patch_size_score):
    """Calculate Habitat Intactness score from edge distance and patch size.

    Args:
        edge_distance_score: Float average edge distance, scaled to 0-1
        patch_size_score: Float average patch size, scaled to 0-1

    Returns:
        Float intactness score (0-1), geometric mean of scaled edge distance and patch size scores
    """
    intactness_score = sqrt(edge_distance_score * patch_size_score)  # Geometric mean
    return intactness_score


def calc_habitat_condition_score(habitat_extent_score, habitat_intactness_score):
    """Calculate overall Habitat Condition score from Habitat Extent and Habitat Intactness.

    Args:
        habitat_extent_score: Float proportion of site that is habitat (0-1)
        habitat_intactness_score: Float intactness score (0-1)

    Returns:
        Float habitat condition score (0-1), product of extent and intactness
    """
    return habitat_extent_score * habitat_intactness_score


def get_habitat_loss_raster(GLC_processed, GPW_processed, HGFC_processed, start_yr):
    """Create habitat loss and start-year habitat rasters.

    Args:
        GLC_processed: ee.Image with processed Global Land Cover data
        GPW_processed: ee.Image with processed Global Pasture Watch data
        HGFC_processed: ee.Image with processed Hansen Global Forest Change data
        start_yr: Integer year to start analysis

    Returns:
        Tuple of ee.Image rasters: (habitat_loss_raster, habitat_start_raster)
    """
    # Create start year habitat binary
    GLC_start = GLC_processed.select(f"GLC_{start_yr}")
    GPW_start = GPW_processed.select(f"GPW_{start_yr}")
    lc_start = GLC_start.where(GPW_start.eq(1), 37)
    anthro_classes = ee.List([1, 2, 3, 4, 30, 37])
    habitat_start = lc_start.remap(
        anthro_classes, ee.List.repeat(0, anthro_classes.size()), defaultValue=1
    )

    # Create end year anthro binary
    forest_loss_binary = HGFC_processed.gt(0)
    GLC_end = GLC_processed.select(f"GLC_{ANALYSIS_END_YR}")
    GPW_end = GPW_processed.select(f"GPW_{ANALYSIS_END_YR}")
    lc_end = GLC_end.where(GPW_end.eq(1), 37)
    anthro_end = lc_end.remap(
        anthro_classes, ee.List.repeat(1, anthro_classes.size()), defaultValue=0
    ).where(forest_loss_binary, 1)

    # Create habitat loss binary
    habitat_loss_binary = habitat_start.And(anthro_end)

    # De-noise (open) habitat loss binary
    habitat_loss_opened = habitat_loss_binary.focalMin(
        radius=OPENING_RADIUS_LOSS, kernelType="square", units="meters"
    ).focalMax(radius=OPENING_RADIUS_LOSS, kernelType="square", units="meters")

    return habitat_loss_opened, habitat_start


def calc_habitat_loss_score(habitat_loss_raster, habitat_start_raster, site_geom):
    """Calculate habitat loss score (proportion of START_YR habitat not lost).

    Args:
        habitat_loss_raster: ee.Image binary raster of habitat loss
        habitat_start_raster: ee.Image binary raster of habitat at start year
        site_geom: ee.Geometry of the PA

    Returns:
        Float habitat loss score (0-1), where 1 = no habitat loss
    """
    habitat_loss_area = (
        ee.Image.pixelArea()
        .updateMask(habitat_loss_raster)
        .reduceRegion(
            ee.Reducer.sum(), site_geom, scale=SCALE, crs=CRS, maxPixels=MAX_PIXELS
        )
        .get("area")
        .getInfo()
    )
    habitat_start_area = (
        ee.Image.pixelArea()
        .updateMask(habitat_start_raster)
        .reduceRegion(
            ee.Reducer.sum(), site_geom, scale=SCALE, crs=CRS, maxPixels=MAX_PIXELS
        )
        .get("area")
        .getInfo()
    )
    if not habitat_start_area:
        return 0
    habitat_loss_proportion = min(habitat_loss_area / habitat_start_area, 1)
    habitat_loss_score = 1 - habitat_loss_proportion
    return habitat_loss_score


def calc_class_area_and_pct(class_image, site_geom, top_n=4):
    """
    Calculate area and percent area for each class in a classified image.

    Args:
        class_image: ee.Image with a single band containing class values
        site_geom: ee.Geometry of the PA
        top_n: Integer to filter to top N classes by area (default: 4)

    Returns:
        ee.Dictionary with keys like '{class}_area' and '{class}_pct'
    """
    # Get class name from the image band name
    class_name = class_image.bandNames().get(0)

    # Calculate area by class
    area_by_class = (
        ee.Image.pixelArea()
        .divide(1000000)  # Get pixel area in sq km
        .addBands(class_image)
        .reduceRegion(
            reducer=ee.Reducer.sum().group(
                groupField=1,
                groupName=class_name,
            ),
            geometry=site_geom,
            scale=SCALE,
            crs=CRS,
            maxPixels=MAX_PIXELS,
        )
    )

    # Convert list to dictionary
    def dict_from_list(item, acc):
        item = ee.Dictionary(item)
        key = item.get(class_name)
        value = item.get("sum")
        return ee.Dictionary(acc).set(key, value)

    class_dict = ee.Dictionary(
        ee.List(area_by_class.get("groups")).iterate(dict_from_list, ee.Dictionary({}))
    )

    # Filter to top N classes by area
    class_dict = class_dict.select(
        class_dict.keys().sort(class_dict.values()).reverse().slice(0, top_n)
    )

    # Append '_area' suffix to keys
    new_keys = class_dict.keys().map(lambda key: ee.String(key).cat("_area"))
    class_dict = ee.Dictionary.fromLists(new_keys, class_dict.values())

    # Add percentage calculations
    site_area = site_geom.area().divide(1000000).getInfo()

    def add_pct_area(key, value):
        pct_key = ee.String(key).slice(0, -5).cat("_pct")
        pct_value = ee.Number(value).divide(site_area).multiply(100)
        return ee.Dictionary().set(key, value).set(pct_key, pct_value)

    def combine_dicts(item, acc):
        return ee.Dictionary(acc).combine(ee.Dictionary(item), True)

    result_dict = ee.Dictionary(
        class_dict.map(add_pct_area).values().iterate(combine_dicts, ee.Dictionary({}))
    )

    return result_dict.getInfo()


def get_driver_class_image(GLC_processed, GPW_processed, habitat_loss_raster):
    """Create classified image of the 4 drivers of habitat loss.

    Args:
        GLC_processed: ee.Image with processed Global Land Cover data
        GPW_processed: ee.Image with processed Global Pasture Watch data
        habitat_loss_raster: ee.Image binary raster of habitat loss

    Returns:
        ee.Image with driver classes: 1=cropland, 2=built-up, 3=pasture, 4=deforestation without conversion
    """
    GLC_end = GLC_processed.select(f"GLC_{ANALYSIS_END_YR}")
    GPW_end = GPW_processed.select(f"GPW_{ANALYSIS_END_YR}")
    lc_end = GLC_end.where(GPW_end.eq(1), 37)
    # Remap landcover classes to driver categories:
    # 1-4 -> 1 (cropland)
    # 30 -> 2 (built-up)
    # 37 -> 3 (pasture)
    # everything else -> 4 (deforestation without conversion)
    driver_class = (
        lc_end.updateMask(habitat_loss_raster)
        .remap([1, 2, 3, 4, 30, 37], [1, 1, 1, 1, 2, 3], defaultValue=4)
        .rename("driver_class")
    )
    return driver_class


def get_habitat_class_image(GLC_processed, habitat_loss_raster, start_yr):
    """Create classified image of habitat types that were lost.

    Args:
        GLC_processed: ee.Image with processed Global Land Cover data
        habitat_loss_raster: ee.Image binary raster of habitat loss
        start_yr: Integer year to start analysis

    Returns:
        ee.Image with habitat class values from start year, masked to loss areas
    """
    GLC_start = GLC_processed.select(f"GLC_{start_yr}")
    habitat_class = GLC_start.updateMask(habitat_loss_raster).rename("habitat_class")
    return habitat_class


def translate_results(results_dict, labels):
    """Print class metrics with human-readable labels and metrics."""
    normalized = {}

    for key, value in results_dict.items():
        if "_" not in key:
            continue

        raw_class_id, metric = key.rsplit("_", 1)
        if metric not in {"area", "pct"}:
            continue

        try:
            class_id = int(float(raw_class_id))
        except TypeError, ValueError:
            continue

        normalized.setdefault(class_id, {})[metric] = value

    # Show classes in descending area
    sorted_class_ids = sorted(
        normalized.keys(),
        key=lambda class_id: normalized[class_id].get("area", 0),
        reverse=True,
    )

    for class_id in sorted_class_ids:
        label = labels.get(class_id, f"Class {class_id}")
        area = normalized[class_id].get("area")
        pct = normalized[class_id].get("pct")

        area_text = f"{area:.2f} km2" if area is not None else "N/A km2"
        pct_text = (
            f"{pct:.2f}% of total PA area"
            if pct is not None
            else "N/A of total PA area"
        )
        print(f"{label}: {area_text}, {pct_text}")


def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image.

    Args:
        image: ee.Image, a Sentinel-2 image

    Returns:
        ee.Image, a cloud-masked Sentinel-2 image
    """
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).divide(10000)


def get_s2_med_composite(site_geom, year):
    """Create median Sentinel-2 composite for a given year and site.
       (For visualization purposes only)

    Args:
        site_geom: ee.Geometry of the PA
        year: Integer year for the composite

    Returns:
        ee.Image median composite of cloud-masked Sentinel-2 imagery
    """
    med_composite = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(site_geom)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .median()
    )
    return med_composite
