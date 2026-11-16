"""
Creates a set of treatment cells for each site.
Treatment cells are 1km x 1km cells that are fully within the site's geometry, on the
covariate stack's 1 km pixel grid, and mostly land (land_frac >= MIN_LAND_FRACTION).
Each site gets a simple random sample of its land cells, sized so that every site's
treatment-cell mean has the same standard error.
Sites with fewer than MIN_TREATMENT_CELLS land cells are dropped (recorded in
DROPPED_SITES), and small sites' samples are raised to TREATMENT_SAMPLE_FLOOR cells.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from cells.get_control_cells import init_ee
from utils.variables import (
    PROJECT,
    EE_CRS_METERS,
    SITE_GEOMS,
    TREATMENT_CELLS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    CELL_SIZE_M,
    RAND_SEED,
    TREATMENT_SE_TARGET,
    MIN_TREATMENT_CELLS,
    TREATMENT_SAMPLE_FLOOR,
    MIN_LAND_FRACTION,
    EE_N_WORKERS,
)
from covariates.export_covariate_stack import load_covariate_stack
from sites.dropped_sites import record_dropped
from utils.retries import report_failures, with_retries

# Largest tile (pixels a side) of the land mask fetched per request
LAND_TILE_PX = 4096


def draw_grid(site_geom, cell_size_m):
    """
    Create a grid of all cells fully within a site geometry, with edges on multiples of
    cell_size_m (the covariate stack's pixel edges).
    """
    # Draw a grid from the site's bounding box, snapped to the pixel grid
    minx, miny, maxx, maxy = site_geom.bounds
    minx = np.floor(minx / cell_size_m) * cell_size_m
    miny = np.floor(miny / cell_size_m) * cell_size_m

    xs, ys = np.meshgrid(
        np.arange(minx, maxx, cell_size_m),
        np.arange(miny, maxy, cell_size_m),
        indexing="ij",
    )
    xs, ys = xs.ravel(), ys.ravel()
    cells = shapely.box(xs, ys, xs + cell_size_m, ys + cell_size_m)

    # Keep cells fully within the site
    site_boundary = site_geom.boundary
    shapely.prepare(site_geom)
    shapely.prepare(site_boundary)
    keep = ~shapely.intersects(site_boundary, cells) & ~shapely.disjoint(
        site_geom, cells
    )

    return gpd.GeoDataFrame(
        geometry=cells[keep], index=np.flatnonzero(keep), crs=GPD_CRS_METERS
    )


def calc_n_treatment(
    n_land_cells, se_target=TREATMENT_SE_TARGET, floor=TREATMENT_SAMPLE_FLOOR
):
    """
    Calculate treatment sample size using the Yamane formula, raised to floor (or to
    every cell, if there are fewer than floor).
    """
    if n_land_cells <= 0:
        return 0
    n = int(np.ceil(n_land_cells / (1 + n_land_cells * se_target**2)))
    return max(n, min(floor, n_land_cells))


def land_mask():
    """1 where a stack pixel is mostly land, else 0 (server side)."""
    land_frac = load_covariate_stack().select("land_frac")
    return land_frac.gte(MIN_LAND_FRACTION).unmask(0).toByte().rename("land")


def fetch_land(land_image, x_lo, y_lo, width, height, label, cell_size_m=CELL_SIZE_M):
    """Land mask pixels (rows from the top) of the width x height cells whose lower-left
    cell is (x_lo, y_lo), in cell units."""
    grid = {
        "dimensions": {"width": int(width), "height": int(height)},
        "affineTransform": {
            "scaleX": cell_size_m,
            "shearX": 0,
            "translateX": float(x_lo * cell_size_m),
            "shearY": 0,
            "scaleY": -cell_size_m,
            "translateY": float((y_lo + height) * cell_size_m),
        },
        "crsWkt": EE_CRS_METERS,
    }
    request = {"expression": land_image, "fileFormat": "NUMPY_NDARRAY", "grid": grid}
    return with_retries(ee.data.computePixels, request, label=label)["land"]


def is_land(cells, land_image, label, cell_size_m=CELL_SIZE_M, tile_px=LAND_TILE_PX):
    """True for each cell that is mostly land, read from the land mask in tiles of up to
    tile_px cells a side (only tiles that hold cells are fetched)."""
    bounds = cells.geometry.bounds
    cx = np.round(bounds["minx"].to_numpy() / cell_size_m).astype(np.int64)
    cy = np.round(bounds["miny"].to_numpy() / cell_size_m).astype(np.int64)
    land = np.zeros(len(cells), dtype=bool)
    tiles = pd.DataFrame({"tx": cx // tile_px, "ty": cy // tile_px})
    for idx in tiles.groupby(["tx", "ty"]).indices.values():
        x_lo, y_lo = cx[idx].min(), cy[idx].min()
        width, height = cx[idx].max() - x_lo + 1, cy[idx].max() - y_lo + 1
        pixels = fetch_land(land_image, x_lo, y_lo, width, height, label, cell_size_m)
        land[idx] = pixels[y_lo + height - 1 - cy[idx], cx[idx] - x_lo] == 1
    return land


def site_treatment_cells(site_id, site_geom, land_image):
    """A site's sampled land treatment cells, or why it was dropped (str)."""
    cells = draw_grid(site_geom, CELL_SIZE_M)
    # Skip the land lookup for sites with too few cells even if all were land
    if len(cells) >= MIN_TREATMENT_CELLS:
        cells = cells[is_land(cells, land_image, label=f"Site {site_id}")]
    if len(cells) < MIN_TREATMENT_CELLS:
        return (
            f"fewer than {MIN_TREATMENT_CELLS} land treatment cells available "
            f"({len(cells)})"
        )
    n_target = calc_n_treatment(len(cells))
    if len(cells) > n_target:
        cells = cells.sample(n=n_target, random_state=RAND_SEED)
    cells["site_id"] = site_id
    cells["protected"] = 1
    return cells


def get_treatment_cells(
    sites_path, output_parquet=TREATMENT_CELLS, n_workers=EE_N_WORKERS
):
    """
    Get each site's treatment cells (a random sample of its land cells, sized by
    calc_n_treatment), n_workers sites at a time.
    """
    init_ee(PROJECT)
    # Read in sites and convert to 6933
    sites_gdf = gpd.read_parquet(sites_path).to_crs(GPD_CRS_METERS)
    land_image = land_mask()

    def cells_or_error(site_id, site_geom):
        try:
            return site_treatment_cells(site_id, site_geom, land_image)
        except Exception as exc:
            return exc

    site_ids = sites_gdf["site_id"].astype(str)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(cells_or_error, site_ids, sites_gdf.geometry))

    all_cells = []
    failed, dropped = {}, {}
    for site_id, result in zip(site_ids, results):
        if isinstance(result, Exception):
            print(f"FAILED: site {site_id}: {result}")
            failed[site_id] = str(result)
        elif isinstance(result, str):
            print(f"Dropped site {site_id}: {result}")
            dropped[site_id] = result
        else:
            all_cells.append(result)

    record_dropped("treatment_cells", dropped)
    report_failures(failed, len(sites_gdf))
    if failed:
        print("Re-run to retry them")
    if not all_cells:
        raise RuntimeError("No treatment cells generated for any site.")

    all_cells = gpd.GeoDataFrame(
        pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
    )
    all_cells["geometry"] = all_cells.geometry.set_precision(1.0)
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    get_treatment_cells(SITE_GEOMS)
