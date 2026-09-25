"""
Creates a set of control cells for each PA.
Control cells are 1km x 1km cells that fall within a given distance range around the PA.
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
    TREATMENT_CELLS,
    CONTROL_CELLS,
    EE_CRS_METERS,
    GPD_CRS_METERS,
    GPD_CRS_PARQUET,
    PSM_CELL_SIZE,
    RAND_SEED,
    CONTROL_INNER_BUFFER,
    CONTROL_OUTER_BUFFER,
    CONTROL_SPACING,
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
    """Count treatment cells per PA from treatment_cells.parquet."""
    treat_df = gpd.read_parquet(treatment_cells_path)
    if treat_df.empty or "WDPAID" not in treat_df.columns:
        return {}
    counts = treat_df.groupby(treat_df["WDPAID"].astype(str)).size()
    return {wdpaid: int(n) for wdpaid, n in counts.items()}


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
    wdpaid: str,
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
            scale=sample_scale_m,
            projection=EE_CRS_METERS,
            seed=seed,
            geometries=True,
        )
    )

    # Set WDPAID as a property of each point
    return points.map(lambda f: f.set("WDPAID", wdpaid))


def points_to_cells(points_fc):
    """
    Draw a cell around each point.
    """
    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame.from_features(
        points_fc.getInfo()["features"], crs="EPSG:4326"
    )

    # Reproject to meter-based CRS for cell construction
    points_gdf = points_gdf.to_crs(GPD_CRS_METERS)

    # Draw a cell around each point
    cell_size = float(PSM_CELL_SIZE)
    half = cell_size / 2.0

    cells = []
    wdpaids = []
    for _, row in points_gdf.iterrows():
        x, y = float(row.geometry.x), float(row.geometry.y)
        cell_geom = box(x - half, y - half, x + half, y + half)
        cells.append(cell_geom)
        wdpaids.append(str(row.get("WDPAID")))

    cells_gdf = gpd.GeoDataFrame(
        {"geometry": cells, "WDPAID": wdpaids}, crs=points_gdf.crs
    )
    cells_gdf["geometry"] = cells_gdf.geometry.set_precision(1.0)
    cells_gdf = cells_gdf.drop_duplicates(subset="geometry")
    cells_gdf["protected"] = 0
    return cells_gdf


def get_control_cells(
    test_sites: str,
    *,
    output_parquet: str = CONTROL_CELLS,
    treatment_cells_path: str = TREATMENT_CELLS,
    n_samples: int | None = None,
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
    try:
        treat_counts = load_treatment_cell_counts(treatment_cells_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Treatment cells not found at {treatment_cells_path}. "
            "Run get_treatment_cells.py first."
        ) from exc

    all_cells = []
    pa_count = 0
    total_pas = len(pa_gdf)
    for _, row in pa_gdf.iterrows():
        wdpaid = int(row["WDPAID"])
        n_treat = treat_counts.get(str(wdpaid), 0)
        if n_treat == 0:
            print(f"Skipping PA: {wdpaid} (0 treatment cells)")
            continue
        n_samples_pa = (
            n_samples if n_samples is not None else calc_n_control_samples(n_treat)
        )
        print("Starting PA: ", wdpaid)
        print(
            f"Treatment cells: {n_treat}; requesting {n_samples_pa} control cells "
            f"(min={CONTROL_N_SAMPLES_MIN}, max={CONTROL_N_SAMPLES_MAX}, "
            f"per_treat={CONTROL_SAMPLES_PER_TREAT})"
        )

        # Use the cleaned site geometry (same one used for treatment cells)
        pa_geom = ee.Geometry(row.geometry.__geo_interface__)

        print("Sampling points for PA: ", wdpaid)
        points_fc = sample_points(
            all_pas,
            pa_geom,
            wdpaid,
            n_samples=n_samples_pa,
            sample_scale_m=sample_scale_m,
            seed=seed,
            inner_buffer_m=inner_buffer_m,
            outer_buffer_m=outer_buffer_m,
        )
        print("Drawing cells for PA: ", wdpaid)
        cells_gdf = points_to_cells(points_fc)

        if len(cells_gdf) == 0:
            print(f"Warning: WDPAID {wdpaid}: no control cells")
            continue

        print(f"Got {len(cells_gdf)} control cells (requested {n_samples_pa})")
        all_cells.append(cells_gdf)

        print("Completed PA: ", wdpaid)
        pa_count += 1
        print(f"Progress: {pa_count}/{total_pas} PAs processed")
        print("--------------------------------")

    if not all_cells:
        raise RuntimeError("No control cells generated for any site.")

    print("Concatenating cells for all PAs")
    all_cells = gpd.GeoDataFrame(
        pd.concat(all_cells, ignore_index=True), crs=GPD_CRS_METERS
    )
    # A cell can be a control for more than one PA, so only drop duplicates within a PA
    all_cells = all_cells.drop_duplicates(subset=["WDPAID", "geometry"])
    all_cells = all_cells.to_crs(GPD_CRS_PARQUET)
    print("Saving cells to parquet: ", output_parquet)
    all_cells.to_parquet(output_parquet)


if __name__ == "__main__":
    get_control_cells(TEST_SITES_GEOJSON)
