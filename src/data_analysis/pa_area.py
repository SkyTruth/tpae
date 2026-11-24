"""
Area of a set of polygons in longitude and latitude, flattened so overlaps count once,
as geodesic area on the WGS84 ellipsoid (the area Protected Planet reports). Flattening
runs in parallel on a TILE_DEG grid; areas add up across tiles.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import shapely
from pyproj import Geod

# Grid cell size (degrees) for flattening
TILE_DEG = 1
# Tiles sent to a worker at once
TILES_PER_TASK = 8
GEOD = Geod(ellps="WGS84")
# shapely geometry type IDs
POLYGON, MULTIPOLYGON, COLLECTION = 3, 6, 7


def polygons(geoms):
    """Polygons in an array of geometries, taken out of multi-part geometries and
    collections; other geometry types are dropped."""
    parts = shapely.get_parts(np.asarray(geoms))
    while (
        multi := np.isin(shapely.get_type_id(parts), [MULTIPOLYGON, COLLECTION])
    ).any():
        parts = np.concatenate([parts[~multi], shapely.get_parts(parts[multi])])
    return parts[shapely.get_type_id(parts) == POLYGON]


def geodesic_area_km2(geom):
    """Geodesic area (km²) on the WGS84 ellipsoid of a geometry's polygons."""
    rings = shapely.orient_polygons(polygons([geom]))
    return sum(GEOD.geometry_area_perimeter(p)[0] for p in rings) / 1e6


def tiles_area_km2(tiles):
    """Total geodesic area (km²) of each tile's shapes flattened within the tile.
    tiles: (tile, shapes) pairs."""
    total = 0.0
    for tile, shapes in tiles:
        pieces = polygons(shapely.intersection(shapes, tile))
        if len(pieces):
            total += geodesic_area_km2(shapely.union_all(pieces))
    return total


def flat_area_km2(shapes, tile_deg=TILE_DEG):
    """Geodesic area (km²) of the union of shapes (overlaps counted once), flattened in
    parallel on a tile_deg grid."""
    shapes = polygons(shapes)
    x0, y0 = [
        a.ravel()
        for a in np.meshgrid(
            np.arange(-180, 180, tile_deg), np.arange(-90, 90, tile_deg)
        )
    ]
    grid = shapely.box(x0, y0, x0 + tile_deg, y0 + tile_deg)
    # Shapes whose bounding boxes overlap each tile
    tile_idx, shape_idx = shapely.STRtree(shapes).query(grid)
    order = np.argsort(tile_idx, kind="stable")
    tile_idx, shape_idx = tile_idx[order], shape_idx[order]
    starts = np.flatnonzero(np.r_[True, tile_idx[1:] != tile_idx[:-1]])
    tiles = [
        (grid[t], shapes[idx])
        for t, idx in zip(tile_idx[starts], np.split(shape_idx, starts[1:]))
    ]
    tasks = [
        tiles[i : i + TILES_PER_TASK] for i in range(0, len(tiles), TILES_PER_TASK)
    ]
    with ProcessPoolExecutor() as executor:
        return sum(executor.map(tiles_area_km2, tasks))
