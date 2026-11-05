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


def extract_cells_with_covariates(cells_fc, covariates, ee_crs_1km, n_cells):
    """Aggregate covariates within cells and join country / ecoregion / biome.

    n_cells: number of cells in cells_fc.
    """
    if n_cells == 0:
        return cells_fc, pd.DataFrame()

    reduced_chunks = []
    df_chunks = []
    for start in range(0, n_cells, EE_GETINFO_LIMIT):
        chunk = cells_fc.filter(
            ee.Filter.And(
                ee.Filter.gte("cell_ID", start),
                ee.Filter.lt("cell_ID", start + EE_GETINFO_LIMIT),
            )
        )
        n_chunk = min(EE_GETINFO_LIMIT, n_cells - start)
        reduced, cells_df = _extract_chunk(chunk, covariates, ee_crs_1km, n_chunk)
        reduced_chunks.append(reduced)
        if len(cells_df) > 0:
            df_chunks.append(cells_df)

    cells_fc = ee.FeatureCollection(reduced_chunks).flatten()
    cells_df = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()

    if len(cells_df) == 0:
        return cells_fc, cells_df

    n_before = len(cells_df)
    n_missing_by_col = cells_df[REQUIRED_COLS].isna().sum()
    cells_df = cells_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_dropped = n_before - len(cells_df)

    if n_dropped > 0:
        print(
            f"⚠ Dropped {n_dropped}/{n_before} cell(s) ({n_dropped / n_before:.1%}) with missing covariate values:"
        )
        for col, n_missing in n_missing_by_col.items():
            if n_missing > 0:
                print(f"    {col}: {n_missing}")

    return cells_fc, cells_df


def _extract_chunk(cells_fc, covariates, ee_crs_1km, n_cells):
    """Run reduceRegions, spatial joins, and getInfo on one <=5000-feature chunk.

    n_cells: number of cells in the chunk, to count cells dropped by the joins.
    """
    cells_fc = covariates.reduceRegions(
        collection=cells_fc,
        reducer=ee.Reducer.mean(),
        scale=CELL_SIZE_M,
        crs=ee_crs_1km,
    ).select(
        "cell_ID",
        *COVARIATES,
        "land_frac",
        "protected",
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
    cells_df = pd.DataFrame([feature["properties"] for feature in cells_list])
    # print(cells_df.head())

    # The joins drop cells with no matching country or ecoregion. Counted here rather
    # than with size().getInfo(), which recomputes the whole chunk on each call
    n_dropped = n_cells - len(cells_df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} cell(s) with no matching country or ecoregion.")

    # Drop any cells that are mostly water
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

    return cells_fc, cells_df
