"""
Build 1 km resampled covariate stack for cell-level PSM.
"""

import ee

from utils.variables import (
    GLO30_ASSET_ID,
    ATC_ASSET_ID,
    POP_ASSET_ID,
    HGFC_ASSET_ID,
)


def build_resampled_covariates(ee_crs_1km):
    """Load covariate images and resample to 1 km resolution."""
    elevation_ic = ee.ImageCollection(GLO30_ASSET_ID).select("DEM")
    elevation = (
        elevation_ic.mosaic()
        .setDefaultProjection(elevation_ic.first().projection())
        .rename("elevation")
    )
    slope = ee.Terrain.slope(elevation).rename("slope")
    treecover2000 = ee.Image(HGFC_ASSET_ID).select("treecover2000")
    travel_time = ee.Image(ATC_ASSET_ID).select("accessibility").rename("travel_time")
    log_pop_density = (
        ee.Image(POP_ASSET_ID)
        .select("population_count")
        .add(1)  # handles zeros for log transform
        .log()
        .rename("log_pop_density")
    )

    def resample(img):
        return (
            img.setDefaultProjection(ee_crs_1km)
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096)
            .reproject(ee_crs_1km)
        )

    elevation = resample(elevation)
    slope = resample(slope)
    treecover2000 = resample(treecover2000)
    travel_time = resample(travel_time)
    log_pop_density = resample(log_pop_density)

    return (
        elevation.addBands(slope)
        .addBands(treecover2000)
        .addBands(travel_time)
        .addBands(log_pop_density)
    )
