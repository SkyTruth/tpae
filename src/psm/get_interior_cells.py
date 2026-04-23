from pathlib import Path
import sys
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Point
from utils.variables import (
    WDPA_TEST_SITE_GEOJSON,
    INTERIOR_CELLS_TEST,
    PSM_CRS,
    PSM_CELL_SIZE,
    RAND_SEED,
    PA_AREA_THRESHOLD,
    SAMPLE_AREA_PCT,
)

def draw_grid(pa_geom, crs, cell_size):
    """
    Create a grid of all valid interior cells within a PA geometry.
    Used for small PAs.
    """
    # Draw a grid from the PA's bounding box
    minx, miny, maxx, maxy = pa_geom.bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=crs)
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
    Randomly sample valid interior cells within a PA geometry.
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
        if any(cell.intersects(existing) and not cell.touches(existing) for existing in cells):
            continue
        cells.append(cell)

    cells = gpd.GeoDataFrame({"geometry": cells}, crs=PSM_CRS)

    return cells



def get_interior_cells(pa_gdf):
    """
    Iterate through a set of PAs and return a set of valid interior cells for each.
    If the PA is less than 500 km2, return a grid of all valid interior cells.
    If the PA is greater than 500 km2, return a random sample of valid interior cells.
    """
    # Read in PAs and convert to 6933
    pa_gdf = gpd.read_file(WDPA_TEST_SITE_GEOJSON)
    pa_gdf = pa_gdf.to_crs(epsg=PSM_CRS)

    all_cells = []

    # Iterate through PAs and get a set of valid interior cells for each
    for _, row in pa_gdf.iterrows():
        pa_geom = row.geometry
        area = pa_geom.area
        # Apply the appropriate function based on the PA's size
        if area < PA_AREA_THRESHOLD:
            cells = draw_grid(pa_geom, PSM_CRS, PSM_CELL_SIZE)
        else:
            cells = sample_cells(pa_geom, (area/1000000) * SAMPLE_AREA_PCT, RAND_SEED, PSM_CELL_SIZE)
        # Add attributes to cells
        cells["WDPA_PID"] = row["WDPA_PID"]
        cells["protected"] = 1
        # Warning for PAs with few or 0 valid cells
        if len(cells) < 50 and len(cells) > 0:
            print(f"Warning: WDPA_PID {row['WDPA_PID']}: only {len(cells)} valid cells")
            continue
        if len(cells) == 0:
            print(f"Warning: WDPA_PID {row['WDPA_PID']}: no valid cells")
            continue
        all_cells.append(cells)

    all_cells = gpd.GeoDataFrame(pd.concat(all_cells, ignore_index=True), crs=PSM_CRS)
    all_cells = all_cells.drop_duplicates(subset="geometry")
    all_cells["geometry"] = all_cells.geometry.set_precision(1.0)
    all_cells = all_cells.to_crs(epsg=4326)
    all_cells.to_parquet(INTERIOR_CELLS_TEST)


if __name__ == "__main__":
    get_interior_cells(WDPA_TEST_SITE_GEOJSON)
