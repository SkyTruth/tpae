import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from pathlib import Path
import sys
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from utils.variables import (
    WDPA_TEST_SITE_GEOJSON,
    WDPA_TEST_SITE_10M_BUFFER,
    WDPA_TEST_SITE_50M_BUFFER,
    WDPA_EXCLUSION_ZONE,
    WDPA_WIDER_LANDSCAPE,
    WDPA_1KM_GRID,
    WDPA_1KM_PSM_GRID,
    PSM_CRS,
)


def make_grids(geom, crs, cell_size=1000, buffer_dist=5000):
    """
    Create 1x1km grids
    """
    aoi = geom.buffer(buffer_dist)
    minx, miny, maxx, maxy = aoi.bounds

    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=crs)

    return grid[grid.intersects(aoi)].copy()


def save_intermediate_gdf(gdf, output_path, save_intermediates):
    if save_intermediates:
        gdf.to_parquet(output_path)


def create_psm_cells(save_intermediates: bool = False):
    # Read in test_site file, convert to 4087
    pa_gdf = gpd.read_file(WDPA_TEST_SITE_GEOJSON)
    pa_gdf = pa_gdf.to_crs(epsg=PSM_CRS)

    # Make Copies for Buffering, to maintain attributes
    pa_gdf_10km_buff = pa_gdf.to_crs(epsg=PSM_CRS).copy()
    pa_gdf_50km_buff = pa_gdf.to_crs(epsg=PSM_CRS).copy()

    # Run the Buffers
    pa_gdf_10km_buff["geometry"] = pa_gdf_10km_buff.buffer(distance=10000)
    pa_gdf_50km_buff["geometry"] = pa_gdf_50km_buff.buffer(distance=50000)

    # Save intermediate files, if specified
    save_intermediate_gdf(
        pa_gdf_10km_buff,
        WDPA_TEST_SITE_10M_BUFFER,
        save_intermediates,
    )
    save_intermediate_gdf(
        pa_gdf_50km_buff,
        WDPA_TEST_SITE_50M_BUFFER,
        save_intermediates,
    )

    # Define the exclusion zone and wider landscapes
    ex_zone = pa_gdf_10km_buff.copy()
    wider_landscape = pa_gdf_50km_buff.copy()
    ex_zone["geometry"] = ex_zone.difference(pa_gdf)
    wider_landscape["geometry"] = wider_landscape.difference(pa_gdf_10km_buff)

    # Save intermediate files, if specified
    save_intermediate_gdf(
        ex_zone,
        WDPA_EXCLUSION_ZONE,
        save_intermediates,
    )
    save_intermediate_gdf(
        wider_landscape,
        WDPA_WIDER_LANDSCAPE,
        save_intermediates,
    )

    poly = pa_gdf_50km_buff
    grids = []

    # Create the grids
    for idx, row in poly.iterrows():
        grid = make_grids(
            row.geometry,
            crs=poly.crs,
            cell_size=1000,  # 1 km cell sizes
            buffer_dist=1000,  # 1 km around each polygon
        )
        grid["WDPA_PID"] = row["WDPA_PID"]  # keep WDPA_PID
        grids.append(grid)

    grid_1km = gpd.GeoDataFrame(pd.concat(grids, ignore_index=True), crs=poly.crs)
    grid_1km = grid_1km.drop_duplicates(subset="geometry")

    # Save intermediate file, if specified
    save_intermediate_gdf(
        grid_1km,
        WDPA_1KM_GRID,
        save_intermediates,
    )

    # Aggregate geometries
    pa_boundary = pa_gdf.union_all().boundary
    ex_geom = ex_zone.union_all()
    outside_geom = pa_gdf_50km_buff.union_all()
    outside_boundary = outside_geom.boundary

    # Stack exclusions
    grid_1km["exclude"] = False
    grid_1km["exclude"] |= grid_1km.intersects(pa_boundary)
    grid_1km["exclude"] |= grid_1km.intersects(ex_geom)
    grid_1km["exclude"] |= grid_1km.intersects(outside_boundary) | grid_1km.disjoint(
        outside_geom
    )

    # Select the valid grid cells for the PA and it's wider landscape
    in_out_grid_1km = grid_1km[~grid_1km["exclude"]]

    pa_union = pa_gdf.union_all()
    in_out_grid_1km = in_out_grid_1km.copy()
    in_out_grid_1km["protected"] = np.where(
        in_out_grid_1km.intersects(pa_union), 1, 0
    ).astype("int8")

    in_out_grid_1km["geometry"] = in_out_grid_1km.geometry.set_precision(
        1.0
    )  # 1 meter precision
    in_out_grid_1km = in_out_grid_1km.drop("exclude", axis=1).to_crs(epsg=4326)
    in_out_grid_1km.to_parquet(WDPA_1KM_PSM_GRID)


if __name__ == "__main__":
    create_psm_cells(save_intermediates=False)
