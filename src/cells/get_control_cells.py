"""
Creates a set of control cells for each site.
Control cells are 1km x 1km cells that fall within a given distance range around the site.
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
    site_id: str,
    *,
    n_samples: int,
    control_spacing_m: int,
    seed: int,
    inner_buffer_m: int,
    outer_buffer_m: int,
):
    """
    Randomly sample unprotected points within a site's buffer zone.
    """
    # Create a donut-shaped buffer zone around the site
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

    # Sample random unprotected points within the donut
    points = (
        ee.Image.constant(0)
        .rename("stratum")
        .updateMask(unprotected_mask)
        .stratifiedSample(
            numPoints=n_samples,
            classBand="stratum",
            region=donut,
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
):
    """
    Iterate through sites and extract control cells for each.
    """
    init_ee(PROJECT)
    sites_gdf = gpd.read_parquet(sites_path)
    # Simplify site boundaries before buffering
    sites_gdf["geometry"] = (
        sites_gdf.geometry.to_crs(GPD_CRS_METERS)
        .simplify(CONTROL_SIMPLIFY_M)
        .to_crs(sites_gdf.crs)
    )
    all_pas = get_all_pas()
    try:
        treat_counts = load_treatment_cell_counts(treatment_cells_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Treatment cells not found at {treatment_cells_path}. "
            "Run get_treatment_cells.py first."
        ) from exc

    all_cells = []
    n_done = 0
    n_sites = len(sites_gdf)
    for _, row in sites_gdf.iterrows():
        site_id = int(row["site_id"])
        n_treat = treat_counts.get(str(site_id), 0)
        if n_treat == 0:
            print(f"Skipping site: {site_id} (0 treatment cells)")
            continue
        n_samples_site = (
            n_samples if n_samples is not None else calc_n_control_samples(n_treat)
        )
        print("Starting site: ", site_id)
        print(
            f"Treatment cells: {n_treat}; requesting {n_samples_site} control cells "
            f"(min={CONTROL_N_SAMPLES_MIN}, max={CONTROL_N_SAMPLES_MAX}, "
            f"per_treat={CONTROL_SAMPLES_PER_TREAT})"
        )

        # Use the cleaned site geometry (same one used for treatment cells)
        site_geom = ee.Geometry(row.geometry.__geo_interface__)

        print("Sampling points for site: ", site_id)
        points_fc = sample_points(
            all_pas,
            site_geom,
            site_id,
            n_samples=n_samples_site,
            control_spacing_m=control_spacing_m,
            seed=seed,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
        )
        print("Drawing cells for site: ", site_id)
        cells_gdf = points_to_cells(points_fc)

        if len(cells_gdf) == 0:
            print(f"Warning: site {site_id}: no control cells")
            continue

        print(f"Got {len(cells_gdf)} control cells (requested {n_samples_site})")
        all_cells.append(cells_gdf)

        print("Completed site: ", site_id)
        n_done += 1
        print(f"Progress: {n_done}/{n_sites} sites processed")
        print("--------------------------------")

    if not all_cells:
        raise RuntimeError("No control cells generated for any site.")

    print("Concatenating cells for all sites")
    all_cells = gpd.GeoDataFrame(
        pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
    )
    # A cell can be a control for more than one site, so only drop duplicates within a site
    all_cells = all_cells.drop_duplicates(subset=["site_id", "geometry"])
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    print("Saving cells to parquet: ", output_parquet)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    get_control_cells(SITE_GEOMS)
