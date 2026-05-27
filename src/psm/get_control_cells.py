"""
Creates a set of control cells for each PA.
Control cells are 1km x 1km cells that are between 10km and 50km from the PA.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from utils.variables import (
    PROJECT,
    PAS_ASSET_ID,
    OECMS_ASSET_ID,
    TEST_SITES_GEOJSON,
    CONTROL_CELLS,
    EE_CRS_METERS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    PSM_CELL_SIZE,
    RAND_SEED,
    CONTROL_INNER_BUFFER,
    CONTROL_OUTER_BUFFER,
    CONTROL_SPACING,
    CONTROL_N_SAMPLES,
)

def init_ee(project):
    """
    Initialize Earth Engine, authenticating if needed.
    """
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def get_all_pas():
    """
    Get a feature collection of all terrestrial PAs and OECMS.
    """
    pas = ee.FeatureCollection(PAS_ASSET_ID)
    oecms = ee.FeatureCollection(OECMS_ASSET_ID)
    return (
        ee.FeatureCollection([pas, oecms])
        .flatten()
        .filter(ee.Filter.eq("REALM", "Terrestrial"))
    )


def sample_points(
    all_pas: ee.FeatureCollection,
    site_geom: ee.Geometry,
    wdpa_pid: str,
    *,
    n_samples: int,
    sample_scale_m: int,
    seed: int,
    inner_buffer_m: int,
    outer_buffer_m: int,
):
    """
    Randomly sample unprotected points within a site's buffer zone.
    """
    # Create a donut-shaped buffer zone around the PA
    buffer_outer = site_geom.buffer(outer_buffer_m)
    buffer_inner = site_geom.buffer(inner_buffer_m)
    donut = buffer_outer.difference(buffer_inner)

    # Mask any protected areas in the donut
    donut_pas = all_pas.filterBounds(donut)
    protected_img = (
        ee.Image(0)
        .byte()
        .paint(donut_pas, 1)
        .rename("protected")
    )
    unprotected_mask = protected_img.unmask(0).eq(0).selfMask()

    # Sample random unprotected points within the donut
    points = (
        ee.Image.constant(0)
        .updateMask(unprotected_mask)
        .sample(
            region=donut,
            scale=sample_scale_m,
            projection=EE_CRS_METERS,
            numPixels=n_samples,
            seed=seed,
            geometries=True,
        )
    )

    # Set WDPA_PID as a property of each point
    points = points.map(lambda f: f.set("WDPA_PID", wdpa_pid))
    return points.limit(n_samples)


def points_to_cells(points_fc):
    """
    Draw a cell around each point.
    """
    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame.from_features(points_fc.getInfo()["features"], crs="EPSG:4326")
    
    # Reproject to meter-based CRS for cell construction
    points_gdf = points_gdf.to_crs(GPD_CRS_METERS)

    # Draw a cell around each point
    cell_size = float(PSM_CELL_SIZE)
    half = cell_size / 2.0

    cells = []
    wdpa_pids = []
    for _, row in points_gdf.iterrows():
        x, y = float(row.geometry.x), float(row.geometry.y)
        cell_geom = box(x - half, y - half, x + half, y + half)
        cells.append(cell_geom)
        wdpa_pids.append(str(row.get("WDPA_PID")))

    cells_gdf = gpd.GeoDataFrame({"geometry": cells, "WDPA_PID": wdpa_pids}, crs=points_gdf.crs)
    cells_gdf["geometry"] = cells_gdf.geometry.set_precision(1.0)
    cells_gdf = cells_gdf.drop_duplicates(subset="geometry")
    cells_gdf["protected"] = 0
    return cells_gdf


def get_control_cells(
    test_sites: str,
    *,
    output_parquet: str = CONTROL_CELLS,
    n_samples: int = CONTROL_N_SAMPLES,
    sample_scale_m: int = CONTROL_SPACING,
    seed: int = RAND_SEED,
    inner_buffer_m: int = CONTROL_INNER_BUFFER,
    outer_buffer_m: int = CONTROL_OUTER_BUFFER,
):
    """
    Iterate through sites and extract control cells for each.
    """
    init_ee(PROJECT)
    pa_gdf = gpd.read_file(test_sites)
    all_pas = get_all_pas()

    all_cells = []
    pa_count = 0
    total_pas = len(pa_gdf)
    for _, row in pa_gdf.iterrows():
        wdpa_pid = int(row["WDPAID"])
        print("Starting PA: ", wdpa_pid)
        
        pa_geom = all_pas.filter(ee.Filter.eq("SITE_ID", wdpa_pid)).geometry()
        
        print("Sampling points for PA: ", wdpa_pid)
        points_fc = sample_points(
            all_pas,
            pa_geom,
            wdpa_pid,
            n_samples=n_samples,
            sample_scale_m=sample_scale_m,
            seed=seed,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
        )
        print("Drawing cells for PA: ", wdpa_pid)
        cells_gdf = points_to_cells(points_fc)
        
        if len(cells_gdf) == 0:
            print(f"Warning: WDPA_PID {wdpa_pid}: no control cells")
            continue
        
        print("Appending cells for PA: ", wdpa_pid)
        all_cells.append(cells_gdf)
        
        print("Completed PA: ", wdpa_pid)
        pa_count += 1
        print(f"Progress: {pa_count}/{total_pas} PAs processed")
        print("--------------------------------")

    if not all_cells:
        raise RuntimeError("No control cells generated for any site.")

    print("Concatenating cells for all PAs")
    all_cells = gpd.GeoDataFrame(pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS)
    all_cells = all_cells.drop_duplicates(subset="geometry")
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    print("Saving cells to parquet: ", output_parquet)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    get_control_cells(TEST_SITES_GEOJSON)
