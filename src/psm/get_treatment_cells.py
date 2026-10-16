"""
Creates a set of treatment cells for each PA.
Treatment cells are 1km x 1km cells that are fully within the PA's geometry.
Each PA gets a simple random sample of its grid of valid interior cells
Sample size is calculated dynamically so that every PA's treatment-cell mean has the same standard error.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from psm.get_control_cells import init_ee
from utils.variables import (
    PROJECT,
    HGFC_ASSET_ID,
    EE_CRS_METERS,
    SCALE,
    MAX_PIXELS,
    TEST_SITES_GEOJSON,
    TREATMENT_CELLS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    PSM_CELL_SIZE,
    RAND_SEED,
    TREATMENT_SE_TARGET,
)


def draw_grid(pa_geom, cell_size):
    """
    Create a grid of all valid treatment cells within a PA geometry.
    """
    # Draw a grid from the PA's bounding box
    minx, miny, maxx, maxy = pa_geom.bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=GPD_CRS_METERS)
    grid = grid.drop_duplicates(subset="geometry")

    # Exclude cells that are not fully within the PA
    pa_boundary = pa_geom.boundary

    grid["exclude"] = False
    grid["exclude"] |= grid.geometry.intersects(pa_boundary)
    grid["exclude"] |= grid.geometry.disjoint(pa_geom)

    valid_grid = grid[~grid["exclude"]]
    valid_grid = valid_grid.drop("exclude", axis=1)

    return valid_grid


def calc_n_treatment(n_land_cells, se_target=TREATMENT_SE_TARGET):
    """
    Calculate treatment sample size so the PA's treatment-cell mean has a standard
    error of se_target standard deviations.
    """
    if n_land_cells <= 0:
        return 0
    n0 = 1 / se_target**2
    return int(np.ceil(n0 / (1 + n0 / n_land_cells)))


def get_land_fraction(pa_geom_4326):
    """Calculate fraction of a PA that is land."""
    land = ee.Image(HGFC_ASSET_ID).select("datamask").eq(1)
    land_frac = (
        land.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry(pa_geom_4326.__geo_interface__),
            scale=SCALE,
            crs=EE_CRS_METERS,
            maxPixels=MAX_PIXELS,
        )
        .get("datamask")
        .getInfo()
    )
    return 1.0 if land_frac is None else land_frac


def get_treatment_cells(test_sites):
    """
    Iterate through a set of PAs and return a set of valid treatment cells for each.
    Each PA's grid of valid interior cells is randomly subsampled to the size given by
    calc_n_treatment (or all cells if the grid is smaller than that).
    """
    init_ee(PROJECT)
    # Read in PAs (EPSG:4326 for Earth Engine) and convert to 6933
    pa_gdf_4326 = gpd.read_file(test_sites)
    pa_gdf = pa_gdf_4326.to_crs(GPD_CRS_METERS)

    all_cells = []

    # Iterate through PAs and get a set of valid treatment cells for each
    for (_, row), geom_4326 in zip(pa_gdf.iterrows(), pa_gdf_4326.geometry):
        pa_geom = row.geometry
        land_frac = get_land_fraction(geom_4326)
        cells = draw_grid(pa_geom, PSM_CELL_SIZE)
        # Size the sample on land cells; water cells are dropped at extraction,
        # so draw extra to keep ~n land cells
        n = calc_n_treatment(len(cells) * land_frac)
        n_draw = int(np.ceil(n / land_frac)) if land_frac > 0 else 0
        if len(cells) > n_draw:
            cells = cells.sample(n=n_draw, random_state=RAND_SEED)
        # Add attributes to cells
        cells["WDPAID"] = str(row.get("WDPAID"))
        cells["protected"] = 1
        # Warning for PAs with few or 0 valid cells
        if len(cells) < 50 and len(cells) > 0:
            print(f"Warning: WDPAID {row['WDPAID']}: only {len(cells)} valid cells")
        if len(cells) == 0:
            print(f"Warning: WDPAID {row['WDPAID']}: no valid cells")
            continue
        all_cells.append(cells)

    all_cells = gpd.GeoDataFrame(
        pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
    )
    all_cells = all_cells.drop_duplicates(subset="geometry")
    all_cells["geometry"] = all_cells.geometry.set_precision(1.0)
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    all_cells.to_parquet(TREATMENT_CELLS)


if __name__ == "__main__":
    get_treatment_cells(TEST_SITES_GEOJSON)
