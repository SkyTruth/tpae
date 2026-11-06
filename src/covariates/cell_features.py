"""
Extract per-cell covariates and spatial attributes for matching.
"""

import ee
import pandas as pd

from utils.variables import (
    COUNTRIES_ASSET_ID,
    BIOME_ASSET_ID,
    CELL_SIZE_M,
    COVARIATES,
    MIN_LAND_FRACTION,
)

REQUIRED_COLS = COVARIATES + [
    "country",
    "ecoregion",
    "biome",
]

# Earth Engine aborts getInfo() after 5000 features.
EE_GETINFO_LIMIT = 5000


def extract_covariates(cells_fc, covariates, ee_crs_1km):
    """Aggregate covariates within cells and join country / ecoregion / biome.

    cells_fc: up to EE_GETINFO_LIMIT cells with a cell_key property. Cells with no
    matching country or ecoregion are dropped.
    """
    cells_fc = covariates.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=CELL_SIZE_M,
        crs=ee_crs_1km,
    ).select(
        "cell_key",
        *COVARIATES,
        "land_frac",
    )

    centroids = cells_fc.map(lambda cell: ee.Feature(cell).centroid())

    countries = ee.FeatureCollection(COUNTRIES_ASSET_ID)
    ecoregions = ee.FeatureCollection(BIOME_ASSET_ID)

    spatial_filter = ee.Filter.intersects(
        leftField=".geo",
        rightField=".geo",
        maxError=1,
    )

    centroids = (
        ee.Join.saveFirst("_match")
        .apply(
            primary=centroids,
            secondary=countries,
            condition=spatial_filter,
        )
        .map(
            lambda f: f.set(
                "country", ee.Feature(f.get("_match")).get("country_na")
            ).set("_match", None)
        )
    )

    centroids = (
        ee.Join.saveFirst("_match")
        .apply(
            primary=centroids,
            secondary=ecoregions,
            condition=spatial_filter,
        )
        .map(
            lambda f: (
                f.set("ecoregion", ee.Feature(f.get("_match")).get("ECO_ID"))
                .set("biome", ee.Feature(f.get("_match")).get("BIOME_NUM"))
                .set("_match", None)
            )
        )
    )

    cells_list = centroids.getInfo()["features"]
    if not cells_list:
        return pd.DataFrame({"cell_key": pd.Series(dtype="int64")})
    return pd.DataFrame([feature["properties"] for feature in cells_list])


def drop_invalid_cells(cells_df):
    """Drop cells that are mostly water or are missing covariate values."""
    is_water = ~(cells_df["land_frac"] >= MIN_LAND_FRACTION)
    if is_water.any():
        n_water_treat = (is_water & (cells_df["protected"] == 1)).sum()
        print(
            f"Dropped {is_water.sum()} cell(s) with land fraction < {MIN_LAND_FRACTION} "
            f"({n_water_treat} treatment, {is_water.sum() - n_water_treat} control)."
        )
    cells_df = cells_df[~is_water].reset_index(drop=True)

    n_before = len(cells_df)
    n_missing_by_col = cells_df[REQUIRED_COLS].isna().sum()
    cells_df = cells_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_dropped = n_before - len(cells_df)

    if n_dropped > 0:
        print(
            f"⚠ Dropped {n_dropped}/{n_before} cell(s) ({n_dropped / n_before:.1%}) with missing covariate values:"
        )
        for col, n in n_missing_by_col.items():
            if n > 0:
                print(f"    {col}: {n}")

    return cells_df
