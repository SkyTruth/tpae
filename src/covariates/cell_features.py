"""
Extract per-cell covariates and spatial attributes for matching.
"""

import ee
import numpy as np
import pandas as pd

from utils.variables import (
    COUNTRIES_ASSET_ID,
    BIOME_ASSET_ID,
    CELL_SIZE_M,
    COVARIATES,
    COVARIATE_STACK_BANDS,
    EE_CRS_METERS,
    GPD_CRS_METERS,
    MIN_LAND_FRACTION,
)

REQUIRED_COLS = COVARIATES + [
    "country",
    "ecoregion",
    "biome",
]
REGION_BANDS = ("country", "ecoregion", "biome")

# Largest window (pixels a side) of the covariate image read per request
TILE_PX = 512
# Value read for missing covariate pixels
NODATA = -9999
# Code of pixels outside every country or ecoregion
NO_REGION = -1
# Largest distance (m) of a cell's corner from the stack's pixel grid
MAX_GRID_OFFSET_M = 1


def covariate_image(covariates):
    """The covariate stack with country, ecoregion and biome codes painted on its 1 km
    grid (server side), and the country name of each country code.

    Each pixel gets the polygon that holds its center, which is the centroid of the
    cell on that pixel.
    """
    countries = ee.FeatureCollection(COUNTRIES_ASSET_ID)
    country_names = sorted(countries.aggregate_array("country_na").distinct().getInfo())
    country_codes = countries.remap(
        country_names, list(range(len(country_names))), "country_na"
    )
    ecoregions = ee.FeatureCollection(BIOME_ASSET_ID)
    regions = [
        ee.Image(NO_REGION).int().paint(country_codes, "country_na").rename("country"),
        ee.Image(NO_REGION).int().paint(ecoregions, "ECO_ID").rename("ecoregion"),
        ee.Image(NO_REGION).int().paint(ecoregions, "BIOME_NUM").rename("biome"),
    ]
    return covariates.unmask(NODATA).addBands(regions).toFloat(), country_names


def stack_pixels(cells):
    """(ix, iy): the covariate stack pixel of each 1 km cell, in CELL_SIZE_M units of
    GPD_CRS_METERS (the stack's CRS)."""
    bounds = cells.geometry.to_crs(GPD_CRS_METERS).bounds
    ix, iy = bounds["minx"] / CELL_SIZE_M, bounds["miny"] / CELL_SIZE_M
    offset_m = max((ix - ix.round()).abs().max(), (iy - iy.round()).abs().max()) * (
        CELL_SIZE_M
    )
    if offset_m > MAX_GRID_OFFSET_M:
        raise ValueError(
            f"Cells are up to {offset_m:.1f} m off the covariate stack's pixel grid"
        )
    return ix.round().astype(int).to_numpy(), iy.round().astype(int).to_numpy()


def extract_covariates(cells, image, country_names):
    """Covariates, country, ecoregion and biome of cells, read from image (from
    covariate_image) in one request.

    cells: cell_key, and ix / iy from stack_pixels, within TILE_PX pixels a side.
    Missing covariates are NaN; cells with no country or ecoregion are dropped.
    """
    x_lo, y_lo = cells["ix"].min(), cells["iy"].min()
    width, height = cells["ix"].max() - x_lo + 1, cells["iy"].max() - y_lo + 1
    grid = {
        "dimensions": {"width": int(width), "height": int(height)},
        "affineTransform": {
            "scaleX": CELL_SIZE_M,
            "shearX": 0,
            "translateX": float(x_lo * CELL_SIZE_M),
            "shearY": 0,
            "scaleY": -CELL_SIZE_M,
            "translateY": float((y_lo + height) * CELL_SIZE_M),
        },
        "crsWkt": EE_CRS_METERS,
    }
    request = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid}
    pixels = ee.data.computePixels(request)
    # Rows count down from the top of the window
    rows = (y_lo + height - 1 - cells["iy"]).to_numpy()
    cols = (cells["ix"] - x_lo).to_numpy()

    values = pd.DataFrame({"cell_key": cells["cell_key"].to_numpy()})
    for band in COVARIATE_STACK_BANDS:
        band_values = pixels[band][rows, cols].astype(float)
        values[band] = np.where(band_values == NODATA, np.nan, band_values)
    for band in REGION_BANDS:
        values[band] = pixels[band][rows, cols].astype(int)
    values = values[
        (values["country"] != NO_REGION) & (values["ecoregion"] != NO_REGION)
    ].reset_index(drop=True)
    values["country"] = [country_names[code] for code in values["country"]]
    return values


def drop_invalid_cells(cells_df):
    """Drop cells that are mostly water or are missing covariate values."""
    is_water = ~(cells_df["land_frac"] >= MIN_LAND_FRACTION)
    if is_water.any():
        n_water_treat = (is_water & (cells_df["protected"] == 1)).sum()
        print(
            f"Dropped {is_water.sum()} cell(s) with land fraction < {MIN_LAND_FRACTION} "
            f"({n_water_treat} treatment, {is_water.sum() - n_water_treat} control)."
        )
    cells_df = cells_df[~is_water].reset_index(drop=True)

    n_before = len(cells_df)
    n_missing_by_col = cells_df[REQUIRED_COLS].isna().sum()
    cells_df = cells_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_dropped = n_before - len(cells_df)

    if n_dropped > 0:
        print(
            f"⚠ Dropped {n_dropped}/{n_before} cell(s) ({n_dropped / n_before:.1%}) with missing covariate values:"
        )
        for col, n in n_missing_by_col.items():
            if n > 0:
                print(f"    {col}: {n}")

    return cells_df
