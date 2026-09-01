"""
Tile grid construction for global propensity sampling.

Builds a regular lon/lat grid covering the globe and filters to tiles
that intersect land. Each tile is identified by integer (i, j) indices
where i increments eastward from -180° and j increments northward from -90°.
"""

from __future__ import annotations

import ee


# Hansen datamask: 1 = land, 2 = permanent water, 0 = no data.
# Using v1.13 (latest as of writing).
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"


def build_tile_grid(
    tile_size_deg: float = 20.0,
    lat_max: float = 84.0,
    lat_min: float = -60.0,
) -> dict[str, ee.Geometry]:
    """
    Build a regular lon/lat grid.

    Excludes Antarctica (below -60°) and the high Arctic above 84° where
    Hansen data ends. Tile keys are 'i_j' strings for stable identifiers.

    Parameters
    ----------
    tile_size_deg : float
        Tile edge length in degrees. Default 20° gives ~160 candidate tiles
        before land filtering.
    lat_max, lat_min : float
        Latitude bounds. Hansen datamask is undefined above ~84° N.

    Returns
    -------
    dict[str, ee.Geometry]
        Mapping from 'i_j' tile ID to ee.Geometry rectangle.
    """
    tiles: dict[str, ee.Geometry] = {}

    n_lon = int(360 / tile_size_deg)
    j = 0
    lat = lat_min
    while lat < lat_max:
        lat_next = min(lat + tile_size_deg, lat_max)
        for i in range(n_lon):
            lon = -180 + i * tile_size_deg
            lon_next = lon + tile_size_deg
            tile_id = f"{i:02d}_{j:02d}"
            tiles[tile_id] = ee.Geometry.Rectangle(
                [lon, lat, lon_next, lat_next],
                proj="EPSG:4326",
                geodesic=False,
            )
        lat = lat_next
        j += 1

    return tiles


def filter_tiles_to_land(
    tiles: dict[str, ee.Geometry],
    coarse_scale: int = 10_000,
    min_land_fraction: float = 0.001,
) -> dict[str, ee.Geometry]:
    """
    Drop tiles with no meaningful land area.

    Uses Hansen datamask reduced at a coarse scale for speed. Computes
    fraction of tile area that is land; drops tiles below threshold.

    Parameters
    ----------
    tiles : dict
        Output of build_tile_grid().
    coarse_scale : int
        Scale (m) for the reduceRegion. 10 km is fast and sufficient to
        detect any tile with non-negligible land.
    min_land_fraction : float
        Minimum fraction of land pixels to keep the tile. 0.001 (~0.1%)
        catches small island tiles while excluding pure ocean.

    Returns
    -------
    dict[str, ee.Geometry]
        Subset of input tiles that contain land.

    Notes
    -----
    This makes one getInfo() per tile sequentially. For ~160 tiles this
    is ~2-3 minutes. Could be parallelized with concurrent.futures but
    it's a one-time setup cost, so simplicity wins.
    """
    is_land = ee.Image(HANSEN_ASSET).select("datamask").eq(1)

    kept: dict[str, ee.Geometry] = {}
    for tile_id, geom in tiles.items():
        stats = is_land.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=coarse_scale,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        land_fraction = stats.get("datamask")
        if land_fraction is not None and land_fraction >= min_land_fraction:
            kept[tile_id] = geom

    return kept


def tiles_to_feature_collection(tiles: dict[str, ee.Geometry]) -> ee.FeatureCollection:
    """
    Convert tile dict to a FeatureCollection for visualization / export.
    """
    features = [
        ee.Feature(geom, {"tile_id": tile_id}) for tile_id, geom in tiles.items()
    ]
    return ee.FeatureCollection(features)
