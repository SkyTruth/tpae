"""
Extract per-cell covariates and spatial attributes for propensity scoring.
"""

import ee
import pandas as pd

from utils.variables import (
    COUNTRIES_ASSET_ID,
    BIOME_ASSET_ID,
    PSM_CELL_SIZE,
    COVARIATES
)

REQUIRED_COLS = COVARIATES + [
    "country",
    "ecoregion",
    "biome",
]


def extract_cells_with_covariates(grid_fc, covariates, ee_crs_1km):
    """Aggregate covariates within grid cells and join country / ecoregion / biome."""
    grid_fc = (
        covariates.reduceRegions(
            collection=grid_fc,
            reducer=ee.Reducer.mean(),
            scale=PSM_CELL_SIZE,
            crs=ee_crs_1km,
        )
        .select(
            "cell_ID",
            *COVARIATES,
            "protected",
        )
    )

    centroids = grid_fc.map(lambda cell: ee.Feature(cell).centroid())

    countries = ee.FeatureCollection(COUNTRIES_ASSET_ID)
    ecoregions = ee.FeatureCollection(BIOME_ASSET_ID)

    spatial_filter = ee.Filter.intersects(
        leftField=".geo",
        rightField=".geo",
        maxError=1,
    )

    centroids = ee.Join.saveFirst("_match").apply(
        primary=centroids,
        secondary=countries,
        condition=spatial_filter,
    ).map(
        lambda f: f.set("country", ee.Feature(f.get("_match")).get("country_na")).set(
            "_match", None
        )
    )

    centroids = ee.Join.saveFirst("_match").apply(
        primary=centroids,
        secondary=ecoregions,
        condition=spatial_filter,
    ).map(
        lambda f: f.set("ecoregion", ee.Feature(f.get("_match")).get("ECO_ID"))
        .set("biome", ee.Feature(f.get("_match")).get("BIOME_NUM"))
        .set("_match", None)
    )

    cells_list = centroids.getInfo()["features"]
    cells_df = pd.DataFrame([feature["properties"] for feature in cells_list])
    # print(cells_df.head())

    n_before = len(cells_df)
    n_missing_by_col = cells_df[REQUIRED_COLS].isna().sum()
    cells_df = cells_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_dropped = n_before - len(cells_df)

    if n_dropped > 0:
        print(
            f"⚠ Dropped {n_dropped}/{n_before} cells ({n_dropped/n_before:.1%}) with missing values:"
        )
        for col, n in n_missing_by_col.items():
            if n > 0:
                print(f"    {col}: {n}")

    if n_dropped == 0:
        print("No cells dropped.")

    return grid_fc, cells_df
