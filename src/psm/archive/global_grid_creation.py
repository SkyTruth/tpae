"""
A rejected approach to PSM grid creation. Given an AOI (country, world, etc.), creates
a globally aligned 1km grid. All cells that are fully within a PA are classified as protected
and retained, and all cells that are fully beyond 10km from a PA are classified as unprotected
and retained.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from pathlib import Path
import sys
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from utils.variables import (
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    PSM_CELL_SIZE,
    CONTROL_INNER_BUFFER,
    PSM_TEST_AOI,
    PSM_TEST_PAS,
    PSM_GLOBAL_GRID,
)


def read_aoi(path: str) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(path)
    aoi = aoi.to_crs(GPD_CRS_METERS)
    aoi = aoi[aoi.geometry.notnull()].copy()
    aoi = aoi.explode(index_parts=False).reset_index(drop=True)
    return aoi


def read_pas(path: str, aoi_union) -> gpd.GeoDataFrame:
    pa_gdf = gpd.read_file(path)
    pa_gdf = pa_gdf.to_crs(GPD_CRS_METERS)
    pa_gdf = pa_gdf[pa_gdf.geometry.notnull()].copy()
    pa_gdf = pa_gdf.explode(index_parts=False).reset_index(drop=True)
    pa_gdf = pa_gdf[pa_gdf.intersects(aoi_union)].copy()
    if pa_gdf.empty:
        raise ValueError("No protected areas intersect the selected AOI.")
    return pa_gdf


def make_grid(aoi_union, cell_size: float) -> gpd.GeoDataFrame:
    """Create a globally aligned 1km square grid over the AOI extent.

    Alignment is anchored to the CRS origin (0, 0), not to the AOI bounds, so
    regions processed separately still share the same lattice.
    """
    minx, miny, maxx, maxy = aoi_union.bounds

    start_x = np.floor(minx / cell_size) * cell_size
    start_y = np.floor(miny / cell_size) * cell_size
    end_x = np.ceil(maxx / cell_size) * cell_size
    end_y = np.ceil(maxy / cell_size) * cell_size

    xs = np.arange(start_x, end_x, cell_size)
    ys = np.arange(start_y, end_y, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({"geometry": cells}, crs=GPD_CRS_METERS)
    grid = grid[grid.intersects(aoi_union)].copy()
    grid.reset_index(drop=True, inplace=True)
    return grid


def add_grid_metadata(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    minx = grid.geometry.bounds.minx.to_numpy()
    miny = grid.geometry.bounds.miny.to_numpy()
    centroids = grid.geometry.centroid

    grid = grid.copy()
    grid["grid_xmin_m"] = minx
    grid["grid_ymin_m"] = miny
    grid["centroid_x_m"] = centroids.x.to_numpy()
    grid["centroid_y_m"] = centroids.y.to_numpy()
    grid["cell_id"] = [f"x{int(round(x))}_y{int(round(y))}" for x, y in zip(minx, miny)]
    return grid


def classify_cells(
    grid: gpd.GeoDataFrame,
    aoi_union,
    pa_union,
    inner_buffer_m: float,
) -> gpd.GeoDataFrame:
    """Identify valid grid cells. Classify cells as 'protected' or 'unprotected' and set protected=1/0.

    protected:   fully within PA union (protected=1)
    unprotected: fully outside PA union buffered by inner_buffer_m (protected=0)
    """
    grid = grid.copy()

    pa_buffer = pa_union.buffer(inner_buffer_m)

    within_aoi = grid.within(aoi_union)
    is_protected = within_aoi & grid.within(pa_union)
    is_unprotected = within_aoi & grid.disjoint(pa_buffer)

    grid_protected = grid[is_protected].copy()
    grid_protected["protected"] = 1

    grid_unprotected = grid[is_unprotected].copy()
    grid_unprotected["protected"] = 0

    valid_grid = pd.concat([grid_protected, grid_unprotected], ignore_index=True)

    return valid_grid


def main() -> None:

    output_path = PSM_GLOBAL_GRID

    print("Reading AOI...")
    aoi_gdf = read_aoi(PSM_TEST_AOI)
    aoi_union = aoi_gdf.union_all()

    print("Reading protected areas intersecting AOI...")
    pa_gdf = read_pas(PSM_TEST_PAS, aoi_union)
    pa_union = pa_gdf.union_all()

    print("Creating globally aligned 1 km grid over AOI extent...")
    grid = make_grid(aoi_union, PSM_CELL_SIZE)
    grid = add_grid_metadata(grid)

    print("Classifying cells as protected / unprotected...")
    grid = classify_cells(grid, aoi_union, pa_union, CONTROL_INNER_BUFFER)

    # 1 m precision
    grid["geometry"] = grid.geometry.set_precision(1.0)

    grid = grid[
        [
            "cell_id",
            "protected",
            "grid_xmin_m",
            "grid_ymin_m",
            "centroid_x_m",
            "centroid_y_m",
            "geometry",
        ]
    ]

    print(f"Exporting {len(grid):,} valid cells to {output_path}...")
    grid = grid.to_crs(GPD_CRS_PARQUET)
    grid.to_parquet(output_path)

    protected_count = int((grid["protected"] == 1).sum())
    unprotected_count = int((grid["protected"] == 0).sum())
    print("Done.")
    print(f"Protected cells: {protected_count:,}")
    print(f"Unprotected cells:   {unprotected_count:,}")


if __name__ == "__main__":
    main()
