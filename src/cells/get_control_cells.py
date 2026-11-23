"""
Creates a set of control cells for each site.
Control cells are 1km x 1km cells that fall within a given distance range around the site.
Sites whose range reaches ±180° longitude (the edges of EASE-Grid) are sampled over a
rectangle on each side of it, masked to the range.
Sites with no control cells, or that fail with an error, are dropped (recorded in
DROPPED_SITES). Use --sites to re-sample only some sites, keeping the others' saved cells.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box

from utils.variables import (
    PROJECT,
    SITE_GEOMS,
    TREATMENT_CELLS,
    CONTROL_CELLS,
    EE_CRS_METERS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    CELL_SIZE_M,
    RAND_SEED,
    CONTROL_INNER_BUFFER_M,
    CONTROL_OUTER_BUFFER_M,
    CONTROL_SIMPLIFY_M,
    CONTROL_SPACING_M,
    CONTROL_N_SAMPLES_MIN,
    CONTROL_N_SAMPLES_MAX,
    CONTROL_SAMPLES_PER_TREAT,
    HGFC_ASSET_ID,
    PAS_ASSET_ID,
    OECMS_ASSET_ID,
    MAB_DESIGNATION,
    EE_N_WORKERS,
)
from sites.dropped_sites import failed_reason, record_dropped
from sites.site_selector import M_PER_DEGREE
from utils.retries import report_failures, set_request_limits, with_retries

# Mean Earth radius (m)
EARTH_RADIUS_M = 6_371_000
# Longitude ranges of the two sides of the antimeridian, ending just short of ±180
SIDES = [(-180 + 1e-6, 0), (0, 180 - 1e-6)]
# Ranges within this factor of the outer buffer of ±180 are split at it
ANTIMERIDIAN_MARGIN = 1.1


def init_ee(project):
    """
    Initialize Earth Engine, authenticating if needed, with set_request_limits.
    """
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    set_request_limits()


def load_treatment_cell_counts(treatment_cells_path: str = TREATMENT_CELLS):
    """Count treatment cells per site from treatment_cells.parquet."""
    treat_df = gpd.read_parquet(treatment_cells_path)
    if treat_df.empty or "site_id" not in treat_df.columns:
        return {}
    counts = treat_df.groupby(treat_df["site_id"].astype(str)).size()
    return {site_id: int(n) for site_id, n in counts.items()}


def calc_n_control_samples(
    n_treat,
    per_treat=CONTROL_SAMPLES_PER_TREAT,
    minimum=CONTROL_N_SAMPLES_MIN,
    maximum=CONTROL_N_SAMPLES_MAX,
):
    """Control sample size: per_treat per treatment cell, clipped to [minimum, maximum]."""
    n_treat = 0 if n_treat is None else int(n_treat)
    return min(maximum, max(minimum, per_treat * n_treat))


def near_antimeridian(geom, distance_m):
    """Whether a shapely geometry (EPSG:4326) comes within distance_m of ±180°."""
    lon, lat = shapely.get_coordinates(shapely.segmentize(geom, 0.05)).T
    gap, lat = np.radians(180 - np.abs(lon)), np.radians(np.abs(lat))
    # Distance to the ±180° meridian, or to the nearest pole if over 90° away from it
    dist = np.where(
        gap < np.pi / 2, np.arcsin(np.sin(gap) * np.cos(lat)), np.pi / 2 - lat
    )
    return dist.min() * EARTH_RADIUS_M < distance_m


def side_rectangles(geom, margin_m):
    """[west, south, east, north] rectangles covering a shapely geometry (EPSG:4326)
    padded by margin_m, one on each side of the antimeridian that the padding reaches."""
    bounds = shapely.bounds(shapely.get_parts(geom))
    pad_lat = margin_m / M_PER_DEGREE
    south = np.maximum(bounds[:, 1] - pad_lat, -90)
    north = np.minimum(bounds[:, 3] + pad_lat, 90)
    max_lat = np.minimum(np.maximum(np.abs(south), np.abs(north)), 89)
    pad_lon = margin_m / (M_PER_DEGREE * np.cos(np.radians(max_lat)))
    west, east = bounds[:, 0] - pad_lon, bounds[:, 2] + pad_lon
    rectangles = []
    for lo, hi in SIDES:
        # Each part's padded extent on this side, including padding past ±180°
        extents = []
        for shift in (-360, 0, 360):
            w, e = np.maximum(west + shift, lo), np.minimum(east + shift, hi)
            on_side = w < e
            extents.append((w[on_side], south[on_side], e[on_side], north[on_side]))
        w, s, e, n = (np.concatenate(col) for col in zip(*extents))
        if len(w):
            rectangles.append([w.min(), s.min(), e.max(), n.max()])
    return rectangles


def sample_points(
    all_pas: ee.FeatureCollection,
    site_geom: ee.Geometry,
    site_id: str,
    *,
    n_samples: int,
    control_spacing_m: int,
    seed: int,
    inner_buffer_m: int,
    outer_buffer_m: int,
    side_rects=None,
):
    """
    Randomly sample unprotected points within a site's buffer zone.

    side_rects: for a buffer zone reaching ±180°, rectangles from side_rectangles to
    sample over instead, masked to the buffer zone cut at ±180°.
    """
    # Create a donut-shaped buffer zone around the site
    buffer_outer = site_geom.buffer(outer_buffer_m)
    buffer_inner = site_geom.buffer(inner_buffer_m)
    donut = buffer_outer.difference(buffer_inner)
    region, donut_mask = donut, None
    if side_rects:
        region = ee.Geometry.MultiPolygon(
            [[[[w, s], [e, s], [e, n], [w, n], [w, s]]] for w, s, e, n in side_rects],
            proj="EPSG:4326",
            geodesic=False,
        )
        # Paint the donut, cut at ±180°, as a mask
        halves = [
            ee.Feature(
                donut.intersection(
                    ee.Geometry.Rectangle([lo, -90, hi, 90], None, False), 1
                )
            )
            for lo, hi in SIDES
        ]
        donut_mask = ee.Image(0).byte().paint(ee.FeatureCollection(halves), 1)

    # Mask any protected areas in the donut
    donut_pas = all_pas.filterBounds(region)
    protected_img = (
        ee.Image(0)
        .byte()
        .paint(donut_pas, 1)
        .rename("protected")
        .unmask(0)
        # Dilate protected areas so control cells are at least 10km from any PA
        .focalMax(radius=10000, kernelType="circle", units="meters")
    )
    unprotected_mask = protected_img.eq(0).selfMask()

    # Apply land mask
    land_mask = (
        ee.Image(HGFC_ASSET_ID)
        .select("datamask")
        .eq(1)  # 1 = land, 2 = permanent water/ocean, 0 = no data
    )
    unprotected_mask = unprotected_mask.updateMask(land_mask)
    if donut_mask is not None:
        unprotected_mask = unprotected_mask.updateMask(donut_mask)

    # Sample random unprotected points within the donut
    points = (
        ee.Image.constant(0)
        .rename("stratum")
        .updateMask(unprotected_mask)
        .stratifiedSample(
            numPoints=n_samples,
            classBand="stratum",
            region=region,
            scale=control_spacing_m,
            projection=EE_CRS_METERS,
            seed=seed,
            geometries=True,
        )
    )

    # Set site_id as a property of each point
    return points.map(lambda f: f.set("site_id", site_id))


def points_to_cells(points_fc):
    """
    Draw a cell around each point.
    """
    # Convert points to GeoDataFrame. computeFeatures pages through the results,
    # so it is not limited to 5000 features like getInfo()
    points_gdf = ee.data.computeFeatures(
        {"expression": points_fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    )

    # If a site has no control points, return an empty GeoDataFrame
    if points_gdf.empty:
        return gpd.GeoDataFrame(
            {"site_id": [], "protected": [], "geometry": []}, crs=GPD_CRS_METERS
        )

    # Reproject to meter-based CRS for cell construction
    points_gdf = points_gdf.set_crs("EPSG:4326").to_crs(GPD_CRS_METERS)

    # Draw a cell around each point
    cell_size_m = float(CELL_SIZE_M)
    half_m = cell_size_m / 2.0

    cells = []
    site_ids = []
    for _, row in points_gdf.iterrows():
        x, y = float(row.geometry.x), float(row.geometry.y)
        cell_geom = box(x - half_m, y - half_m, x + half_m, y + half_m)
        cells.append(cell_geom)
        site_ids.append(str(row.get("site_id")))

    cells_gdf = gpd.GeoDataFrame(
        {"geometry": cells, "site_id": site_ids}, crs=points_gdf.crs
    )
    cells_gdf["geometry"] = cells_gdf.geometry.set_precision(1.0)
    cells_gdf = cells_gdf.drop_duplicates(subset="geometry")
    cells_gdf["protected"] = 0
    return cells_gdf


def get_control_cells(
    sites_path: str,
    *,
    output_parquet: str = CONTROL_CELLS,
    treatment_cells_path: str = TREATMENT_CELLS,
    n_samples: int | None = None,
    control_spacing_m: int = CONTROL_SPACING_M,
    seed: int = RAND_SEED,
    inner_buffer_m: int = CONTROL_INNER_BUFFER_M,
    outer_buffer_m: int = CONTROL_OUTER_BUFFER_M,
    n_workers: int = EE_N_WORKERS,
    site_ids: list[int] | None = None,
):
    """
    Extract control cells for each site, n_workers sites at a time.

    site_ids: only these sites, replacing their saved cells and dropped records.
    """
    init_ee(PROJECT)
    sites_gdf = gpd.read_parquet(sites_path)
    # Simplify site boundaries before buffering
    sites_gdf["geometry"] = (
        sites_gdf.geometry.to_crs(GPD_CRS_METERS)
        .simplify(CONTROL_SIMPLIFY_M)
        .to_crs(sites_gdf.crs)
    )
    # Every PA and OECM in all realms, except UNESCO-MAB Biosphere Reserve PAs
    mab_pas = ee.Filter.And(
        ee.Filter.eq("DESIG_ENG", MAB_DESIGNATION), ee.Filter.eq("SITE_TYPE", "PA")
    )
    all_pas = (
        ee.FeatureCollection(
            [ee.FeatureCollection(PAS_ASSET_ID), ee.FeatureCollection(OECMS_ASSET_ID)]
        )
        .flatten()
        .filter(mab_pas.Not())
    )
    try:
        treat_counts = load_treatment_cell_counts(treatment_cells_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Treatment cells not found at {treatment_cells_path}. "
            "Run get_treatment_cells.py first."
        ) from exc

    # Sites with treatment cells, and the number of control cells to request for each
    site_jobs = []
    for _, row in sites_gdf.iterrows():
        site_id = int(row["site_id"])
        n_treat = treat_counts.get(str(site_id), 0)
        if n_treat == 0:
            continue
        n_samples_site = (
            n_samples if n_samples is not None else calc_n_control_samples(n_treat)
        )
        site_jobs.append((site_id, row.geometry, n_samples_site))
    print(
        f"Skipping {len(sites_gdf) - len(site_jobs)} site(s) with no treatment cells "
        "(dropped at earlier steps)"
    )
    if site_ids is not None:
        site_jobs = [job for job in site_jobs if job[0] in site_ids]
        print(f"Sampling only the listed sites ({len(site_jobs)} with treatment cells)")
    print(
        f"Sampling control cells for {len(site_jobs)} sites, {n_workers} at a time "
        f"(min={CONTROL_N_SAMPLES_MIN}, max={CONTROL_N_SAMPLES_MAX}, "
        f"per_treat={CONTROL_SAMPLES_PER_TREAT})"
    )

    def sample_site(site_id, geom, n_samples_site):
        # Split buffer zones reaching ±180° into a rectangle on each side
        side_rects = None
        if near_antimeridian(geom, outer_buffer_m * ANTIMERIDIAN_MARGIN):
            side_rects = side_rectangles(geom, outer_buffer_m)
        # Use the cleaned site geometry (same one used for treatment cells)
        points_fc = sample_points(
            all_pas,
            ee.Geometry(geom.__geo_interface__),
            site_id,
            n_samples=n_samples_site,
            control_spacing_m=control_spacing_m,
            seed=seed,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
            side_rects=side_rects,
        )
        return points_to_cells(points_fc)

    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = {
            executor.submit(with_retries, sample_site, *job, label=f"Site {job[0]}"): (
                job
            )
            for job in site_jobs
        }
        site_cells, failed = {}, {}
        for n_done, future in enumerate(as_completed(futures), start=1):
            site_id, _, n_samples_site = futures[future]
            try:
                site_cells[site_id] = future.result()
            except Exception as exc:
                print(f"FAILED: site {site_id}: {exc} [{n_done}/{len(site_jobs)}]")
                failed[site_id] = str(exc)
                continue
            n_cells = len(site_cells[site_id])
            print(
                f"Site {site_id}: {n_cells} control cells "
                f"(requested {n_samples_site}) [{n_done}/{len(site_jobs)}]"
            )
            if n_cells == 0:
                print(f"Dropped site {site_id}: no control cells")
    finally:
        # Stop unstarted sites if interrupted
        executor.shutdown(cancel_futures=True)
    report_failures(failed, len(site_jobs))
    if failed:
        print("Re-run to retry them")
    dropped = {}
    for site_id, _, _ in site_jobs:
        if site_id in failed:
            dropped[site_id] = failed_reason(failed[site_id])
        elif len(site_cells[site_id]) == 0:
            dropped[site_id] = "no control cells"
    record_dropped("control_cells", dropped, sites=site_ids)

    # Combine in site order, so the output does not depend on which site finished first
    all_cells = [
        site_cells[job[0]]
        for job in site_jobs
        if job[0] in site_cells and len(site_cells[job[0]]) > 0
    ]

    if all_cells:
        print("Concatenating cells for all sites")
        all_cells = gpd.GeoDataFrame(
            pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
        )
        # A cell can be a control for more than one site, so only drop duplicates within a site
        all_cells = all_cells.drop_duplicates(subset=["site_id", "geometry"])
        all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    elif site_ids is None:
        raise RuntimeError("No control cells generated for any site.")
    if site_ids is not None:
        # Replace the listed sites' saved cells
        saved = gpd.read_parquet(output_parquet)
        saved = saved[~saved["site_id"].astype(int).isin(site_ids)]
        new = [all_cells] if len(all_cells) else []
        all_cells = pd.concat([saved, *new], ignore_index=True)
    print("Saving cells to parquet: ", output_parquet)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites",
        nargs="+",
        type=int,
        metavar="SITE_ID",
        help="re-sample only these sites, keeping the other sites' saved cells",
    )
    get_control_cells(SITE_GEOMS, site_ids=parser.parse_args().sites)
