"""
Country boundary, protected areas, protected land, and treatment and control cells for
national statistics, where all of a country's protected land is analyzed as one site.

Same rules as get_treatment_cells.py, get_control_cells.py and
extract_cell_covariates.py, with the country as the region of every Earth Engine step.
PAs are not clipped to the country.
"""

from concurrent.futures import ThreadPoolExecutor

import ee
import pandas as pd

from cells.get_control_cells import calc_n_control_samples, points_to_cells
from cells.get_treatment_cells import calc_n_treatment, land_mask
from covariates.cell_features import (
    TILE_PX,
    covariate_image,
    drop_invalid_cells,
    extract_covariates,
    stack_pixels,
)
from covariates.export_covariate_stack import load_covariate_stack
from sites.chunker import wdpa_features
from sites.site_selector import bounds_rectangle
from utils.retries import with_retries
from utils.variables import (
    ANALYSIS_START_YR,
    CELL_SIZE_M,
    CONTROL_INNER_BUFFER_M,
    CONTROL_SPACING_M,
    EE_CRS_METERS,
    EE_N_WORKERS,
    HGFC_ASSET_ID,
    MAB_DESIGNATION,
    MAX_PIXELS,
    MIN_TREATMENT_CELLS,
    OECMS_ASSET_ID,
    PAS_ASSET_ID,
    RAND_SEED,
)

# Country boundaries, one feature per country (ISO3 code in COUNTRY_ISO3_PROPERTY)
COUNTRY_BOUNDARIES_ASSET_ID = "WM/geoLab/geoBoundaries/600/ADM0"
COUNTRY_ISO3_PROPERTY = "shapeGroup"
# Pixel size (m) of the protected land used to find cells fully inside it
INTERIOR_SCALE_M = 50
# tileScale of requests that read interior_land_image, to fit its 50 m pixels in memory
INTERIOR_TILE_SCALE = 16
# WDPA attributes downloaded for the PA summary
PA_COLUMNS = [
    "SITE_ID",
    "NAME",
    "ISO3",
    "REALM",
    "STATUS",
    "STATUS_YR",
    "DESIG_ENG",
    "GIS_AREA",
]


def load_country(iso3):
    """A country's boundary: (Earth Engine geometry, shapely geometry in EPSG:4326)."""
    country_fc = ee.FeatureCollection(COUNTRY_BOUNDARIES_ASSET_ID).filter(
        ee.Filter.eq(COUNTRY_ISO3_PROPERTY, iso3)
    )
    country_gdf = with_retries(
        ee.data.computeFeatures,
        {"expression": country_fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"},
        label=f"Country {iso3}",
    )
    if country_gdf.empty:
        raise ValueError(
            f"No country with {COUNTRY_ISO3_PROPERTY} {iso3!r} in "
            f"{COUNTRY_BOUNDARIES_ASSET_ID}"
        )
    return country_fc.geometry(), country_gdf.set_crs("EPSG:4326").union_all()


def national_pas(country_geom):
    """PAs in wdpa_features designated before ANALYSIS_START_YR that intersect the
    country, not clipped to it (server side)."""
    return (
        wdpa_features()
        .filter(ee.Filter.lt("STATUS_YR", ANALYSIS_START_YR))
        .filterBounds(country_geom)
    )


def pa_table(pas):
    """PA_COLUMNS of every PA feature, without geometries."""
    features = pas.select(PA_COLUMNS, None, False)
    table = with_retries(
        ee.data.computeFeatures,
        {"expression": features, "fileFormat": "PANDAS_DATAFRAME"},
        label="PA attributes",
    )
    return table.reindex(columns=PA_COLUMNS)


def protected_land(pas):
    """Every PA dissolved into one image: 1 = protected, 0 = not (server side)."""
    return ee.Image(0).byte().paint(pas, 1).rename("protected")


def interior_land_image(protected):
    """1 for each covariate stack pixel (1 km cell) fully inside protected land and
    mostly land, else 0 (server side)."""
    fine = protected.reproject(ee.Projection(EE_CRS_METERS).atScale(INTERIOR_SCALE_M))
    # A cell is fully protected if every INTERIOR_SCALE_M pixel in it is
    interior = fine.reduceResolution(
        ee.Reducer.min(), maxPixels=(CELL_SIZE_M // INTERIOR_SCALE_M) ** 2
    ).reproject(ee.Projection(EE_CRS_METERS).atScale(CELL_SIZE_M))
    return interior.And(land_mask()).rename("interior")


def count_cells(image, region, scale_m=CELL_SIZE_M, tile_scale=1):
    """Number of pixels of a 0/1 image that are 1, with centers in region."""
    stats = image.reduceRegion(
        ee.Reducer.sum().unweighted(),
        region,
        crs=EE_CRS_METERS,
        scale=scale_m,
        maxPixels=MAX_PIXELS,
        tileScale=tile_scale,
    )
    return round(with_retries(stats.getInfo, label="Cell count").popitem()[1])


def sample_points_in(mask, region, n_samples, scale_m, site_id, tile_scale=1):
    """Up to n_samples random pixel centers on the scale_m grid of EE_CRS_METERS where
    mask is 1, within region, with site_id set (server side)."""
    points = (
        ee.Image.constant(0)
        .rename("stratum")
        .updateMask(mask)
        .stratifiedSample(
            numPoints=n_samples,
            classBand="stratum",
            region=region,
            scale=scale_m,
            projection=EE_CRS_METERS,
            seed=RAND_SEED,
            geometries=True,
            tileScale=tile_scale,
        )
    )
    return points.map(lambda f: f.set("site_id", site_id))


def sample_treatment_cells(country, country_geom, interior):
    """A random sample of the country's interior land cells (from
    interior_land_image), sized by calc_n_treatment, and the number of interior land
    cells."""
    n_interior = count_cells(interior, country_geom, tile_scale=INTERIOR_TILE_SCALE)
    if n_interior < MIN_TREATMENT_CELLS:
        raise ValueError(
            f"{country}: fewer than {MIN_TREATMENT_CELLS} land treatment cells "
            f"available ({n_interior})"
        )
    points = sample_points_in(
        interior,
        country_geom,
        calc_n_treatment(n_interior),
        CELL_SIZE_M,
        country,
        tile_scale=INTERIOR_TILE_SCALE,
    )
    cells = with_retries(points_to_cells, points, label="Treatment cells")
    cells["protected"] = 1
    return cells, n_interior


def control_land_image(country_bounds):
    """1 on land at least CONTROL_INNER_BUFFER_M from any PA or OECM (except UNESCO-MAB
    Biosphere Reserve PAs), masked elsewhere, as in get_control_cells.sample_points
    (server side).

    country_bounds: (min lon, min lat, max lon, max lat); PAs and OECMs within
    CONTROL_INNER_BUFFER_M of these bounds are masked.
    """
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
    nearby_pas = all_pas.filterBounds(
        bounds_rectangle(country_bounds, CONTROL_INNER_BUFFER_M)
    )
    protected_img = (
        ee.Image(0)
        .byte()
        .paint(nearby_pas, 1)
        .rename("protected")
        .unmask(0)
        # Dilate protected areas so control cells are at least 10km from any PA
        .focalMax(radius=CONTROL_INNER_BUFFER_M, kernelType="circle", units="meters")
    )
    land = ee.Image(HGFC_ASSET_ID).select("datamask").eq(1)
    return protected_img.eq(0).selfMask().updateMask(land).rename("control_land")


def sample_control_cells(country, country_geom, country_bounds, n_treat):
    """Control cells sampled anywhere in the country, calc_n_control_samples(n_treat)
    of them, CONTROL_SPACING_M apart."""
    points = sample_points_in(
        control_land_image(country_bounds),
        country_geom,
        calc_n_control_samples(n_treat),
        CONTROL_SPACING_M,
        country,
    )
    return with_retries(points_to_cells, points, label="Control cells")


def extract_country_covariates(cells, n_workers=EE_N_WORKERS):
    """Covariates, country, ecoregion and biome of cells (cell_ID, protected,
    geometry), read from the covariate stack in tiles as extract_cell_covariates.py
    does, with water cells and cells missing values dropped."""
    image, country_names = with_retries(
        covariate_image, load_covariate_stack(), label="Country names"
    )
    cells = cells[["cell_ID", "protected", "geometry"]].assign(
        cell_key=cells["cell_ID"]
    )
    cells["ix"], cells["iy"] = stack_pixels(cells)
    tiles = [
        tile[["cell_key", "ix", "iy"]]
        for _, tile in cells.groupby([cells["ix"] // TILE_PX, cells["iy"] // TILE_PX])
    ]

    def extract_tile(i):
        return with_retries(
            extract_covariates, tiles[i], image, country_names, label=f"Tile {i}"
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        values = pd.concat(executor.map(extract_tile, range(len(tiles))))
    print(f"Extracted {len(cells)} cells in {len(tiles)} tile(s)")

    cells_df = cells[["cell_ID", "protected", "cell_key"]].merge(values, on="cell_key")
    n_unjoined = len(cells) - len(cells_df)
    if n_unjoined > 0:
        print(f"Dropped {n_unjoined} cell(s) with no matching country or ecoregion.")
    cells_df = cells_df.drop(columns="cell_key").sort_values("cell_ID")
    cells_df = drop_invalid_cells(cells_df.reset_index(drop=True))
    # Columns in alphabetical order, as extract_cell_covariates.py saves them
    return cells_df[sorted(cells_df.columns)]
