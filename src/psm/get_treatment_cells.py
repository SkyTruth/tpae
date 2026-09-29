"""
Creates a set of treatment cells for each PA.
Treatment cells are 1km x 1km cells that are fully within the PA's geometry.
If the PA is small, a grid of all valid interior cells is returned.
If the PA is large, a random sample of valid interior cells is returned.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Point
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
    PA_AREA_THRESHOLD,
    SAMPLE_AREA_PCT,
)


def draw_grid(pa_geom, cell_size):
    """
    Create a grid of all valid treatment cells within a PA geometry.
    Used for small PAs.
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


def sample_cells(pa_geom, n_samples, seed, cell_size):
    """
    Randomly sample valid treatment cells within a PA geometry.
    Used for large PAs.
    """
    half = cell_size / 2.0
    minx, miny, maxx, maxy = pa_geom.bounds
    boundary = pa_geom.boundary

    cells = []
    rng = np.random.default_rng(seed)
    max_attempts = max(n_samples * 500, 50_000)
    attempts = 0

    while len(cells) < n_samples and attempts < max_attempts:
        attempts += 1
        # Randomly sample a point within the PA's bounding box
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        point = Point(x, y)
        # Reject the point if it is not within the PA
        if not pa_geom.contains(point):
            continue
        # Draw a 1km x 1km cell around the point
        cell = box(x - half, y - half, x + half, y + half)
        # Reject the cell if it is not fully within the PA
        if cell.disjoint(pa_geom) or cell.intersects(boundary):
            continue
        # Reject the cell if it overlaps any previously accepted cell.
        if any(
            cell.intersects(existing) and not cell.touches(existing)
            for existing in cells
        ):
            continue
        cells.append(cell)

    cells = gpd.GeoDataFrame({"geometry": cells}, crs=GPD_CRS_METERS)

    return cells


def get_land_fraction(pa_geom_4326):
    """Share of a PA that is land (Hansen datamask, same mask used to drop water cells)."""
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
    If the PA has less than 500 km2 of land area, return a grid of all valid interior cells.
    Otherwise, return a random sample of valid interior cells.
    """
    init_ee(PROJECT)
    # Read in PAs (EPSG:4326 for Earth Engine) and convert to 6933
    pa_gdf_4326 = gpd.read_file(test_sites)
    pa_gdf = pa_gdf_4326.to_crs(GPD_CRS_METERS)

    all_cells = []

    # Iterate through PAs and get a set of valid treatment cells for each
    for (_, row), geom_4326 in zip(pa_gdf.iterrows(), pa_gdf_4326.geometry):
        pa_geom = row.geometry
        area = pa_geom.area
        land_area = area * get_land_fraction(geom_4326)
        if land_area < PA_AREA_THRESHOLD:
            cells = draw_grid(pa_geom, PSM_CELL_SIZE)
        else:
            cells = sample_cells(
                pa_geom, (area / 1000000) * SAMPLE_AREA_PCT, RAND_SEED, PSM_CELL_SIZE
            )
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
