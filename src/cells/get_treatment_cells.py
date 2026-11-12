"""
Creates a set of treatment cells for each site.
Treatment cells are 1km x 1km cells that are fully within the site's geometry.
Each site gets a simple random sample of its grid of valid interior cells
Sample size is calculated dynamically so that every site's treatment-cell mean has the same standard error.
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
    HGFC_ASSET_ID,
    EE_CRS_METERS,
    SCALE_M,
    MAX_PIXELS,
    SITE_GEOMS,
    TREATMENT_CELLS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    CELL_SIZE_M,
    RAND_SEED,
    TREATMENT_SE_TARGET,
    EE_N_WORKERS,
)
from utils.retries import report_failures, with_retries


def draw_grid(site_geom, cell_size_m):
    """
    Create a grid of all valid treatment cells within a site geometry.
    """
    # Draw a grid from the site's bounding box
    minx, miny, maxx, maxy = site_geom.bounds

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


def calc_n_treatment(n_land_cells, se_target=TREATMENT_SE_TARGET):
    """
    Calculate treatment sample size using the Yamane formula.
    """
    if n_land_cells <= 0:
        return 0
    return int(np.ceil(n_land_cells / (1 + n_land_cells * se_target**2)))


def get_land_fraction(site_geom_4326):
    """Calculate fraction of a site that is land."""
    land = ee.Image(HGFC_ASSET_ID).select("datamask").eq(1)
    land_frac = (
        land.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry(site_geom_4326.__geo_interface__),
            scale=SCALE_M,
            crs=EE_CRS_METERS,
            maxPixels=MAX_PIXELS,
        )
        .get("datamask")
        .getInfo()
    )
    return 1.0 if land_frac is None else land_frac


def get_treatment_cells(
    sites_path, output_parquet=TREATMENT_CELLS, n_workers=EE_N_WORKERS
):
    """
    Iterate through a set of sites and return a set of valid treatment cells for each.
    Each site's grid of valid interior cells is randomly subsampled to the size given by
    calc_n_treatment (or all cells if the grid is smaller than that).
    """
    init_ee(PROJECT)
    # Read in sites (EPSG:4326 for Earth Engine) and convert to 6933
    sites_gdf_4326 = gpd.read_parquet(sites_path)
    sites_gdf = sites_gdf_4326.to_crs(GPD_CRS_METERS)

    # Land fraction of each site (or the error it raised), n_workers sites at a time
    def land_fraction_or_error(site_id, geom_4326):
        try:
            return with_retries(get_land_fraction, geom_4326, label=f"Site {site_id}")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        land_fracs = list(
            executor.map(
                land_fraction_or_error,
                sites_gdf_4326["site_id"],
                sites_gdf_4326.geometry,
            )
        )

    all_cells = []

    # Iterate through sites and get a set of valid treatment cells for each
    failed = {}
    for (_, row), land_frac in zip(sites_gdf.iterrows(), land_fracs):
        site_geom = row.geometry
        site_id = str(row["site_id"])
        try:
            if isinstance(land_frac, Exception):
                raise land_frac
            cells = draw_grid(site_geom, CELL_SIZE_M)
            # Size the sample on land cells; water cells are dropped at extraction,
            # so draw extra to keep ~n_target land cells
            n_target = calc_n_treatment(len(cells) * land_frac)
            n_draw = int(np.ceil(n_target / land_frac)) if land_frac > 0 else 0
            if len(cells) > n_draw:
                cells = cells.sample(n=n_draw, random_state=RAND_SEED)
        except Exception as exc:
            print(f"FAILED: site {site_id}: {exc}")
            failed[site_id] = str(exc)
            continue
        # Add attributes to cells
        cells["site_id"] = site_id
        cells["protected"] = 1
        # Warning for sites with few or 0 valid cells
        if len(cells) < 50 and len(cells) > 0:
            print(f"Warning: site {site_id}: only {len(cells)} valid cells")
        if len(cells) == 0:
            print(f"Warning: site {site_id}: no valid cells")
            continue
        all_cells.append(cells)

    report_failures(failed, len(sites_gdf))
    if failed:
        print("Re-run to retry them")
    if not all_cells:
        raise RuntimeError("No treatment cells generated for any site.")

    all_cells = gpd.GeoDataFrame(
        pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
    )
    all_cells = all_cells.drop_duplicates(subset="geometry")
    all_cells["geometry"] = all_cells.geometry.set_precision(1.0)
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    get_treatment_cells(SITE_GEOMS)
