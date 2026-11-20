"""
Read the active site group's cleaned sites (SITE_GEOMS, written by clean_sites.py),
reduce Earth Engine images over a site, cut a site into pieces for intactness, and
build simple rectangles to select input images with.
"""

import math

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from utils.variables import SITE_GEOMS, GPD_CRS_METERS, INTACTNESS_CHUNK_KM

M_PER_KM = 1000
# Meters per degree of latitude (and of longitude at the equator), rounded down
M_PER_DEGREE = 110_000


def _read_sites(reader, path, **kwargs):
    try:
        return reader(path, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No cleaned sites at {path}; run clean_sites.py first"
        ) from None


def load_sites(path=SITE_GEOMS):
    """Cleaned sites, one row per site_id (EPSG:4326)."""
    return _read_sites(gpd.read_parquet, path)


def load_start_yrs(path=SITE_GEOMS):
    """{site_id: start year} for every site, without reading geometries."""
    table = _read_sites(pd.read_parquet, path, columns=["site_id", "start_yr"])
    return dict(zip(table["site_id"].astype(int), table["start_yr"].astype(int)))


def load_site_ids(path=SITE_GEOMS):
    """SITE_IDs of every site, in file order."""
    return list(load_start_yrs(path))


def ee_geometry(geom):
    """Earth Engine geometry of a shapely geometry."""
    return ee.Geometry(geom.__geo_interface__)


def padded_bounds(bounds, margin_m=0):
    """[min lon, min lat, max lon, max lat] of bounds, at least margin_m wider on each
    side, within ±180° longitude and ±90° latitude."""
    min_lon, min_lat, max_lon, max_lat = (float(b) for b in bounds)
    pad_lat = margin_m / M_PER_DEGREE
    lat_lo, lat_hi = max(min_lat - pad_lat, -90), min(max_lat + pad_lat, 90)
    max_abs_lat = min(max(abs(lat_lo), abs(lat_hi)), 89)
    pad_lon = margin_m / (M_PER_DEGREE * math.cos(math.radians(max_abs_lat)))
    return [max(min_lon - pad_lon, -180), lat_lo, min(max_lon + pad_lon, 180), lat_hi]


def bounds_rectangle(bounds, margin_m=0):
    """Earth Engine rectangle around bounds (min lon, min lat, max lon, max lat), at
    least margin_m wider on each side."""
    return ee.Geometry.Rectangle(
        padded_bounds(bounds, margin_m), proj="EPSG:4326", geodesic=False
    )


def side_bounds(geoms):
    """Bounds of shapely geometries (EPSG:4326) as a list: their total bounds, or, if
    they span more than 180° of longitude, the bounds of the parts centered west of 0°
    and of those east of it."""
    parts = shapely.get_parts(geoms)
    total = shapely.total_bounds(parts)
    if total[2] - total[0] <= 180:
        return [total]
    bounds = shapely.bounds(parts)
    west = bounds[:, 0] + bounds[:, 2] < 0
    return [shapely.total_bounds(parts[side]) for side in (west, ~west) if side.any()]


def bounds_region(geoms, margin_m=0):
    """bounds_rectangle around shapely geometries (EPSG:4326), or a multipolygon of one
    rectangle on each side of the antimeridian if they span more than 180° of
    longitude."""
    sides = side_bounds(geoms)
    if len(sides) == 1:
        return bounds_rectangle(sides[0], margin_m)
    polygons = []
    for bounds in sides:
        w, s, e, n = padded_bounds(bounds, margin_m)
        polygons.append([[[w, s], [e, s], [e, n], [w, n], [w, s]]])
    return ee.Geometry.MultiPolygon(polygons, proj="EPSG:4326", geodesic=False)


def site_region(geom):
    """A site for reduce_site: its polygons as an Earth Engine FeatureCollection, and
    its bounding box from bounds_region."""
    parts = getattr(geom, "geoms", [geom])
    site_fc = ee.FeatureCollection([ee.Feature(ee_geometry(p)) for p in parts])
    return site_fc, bounds_region(geom)


def reduce_site(image, reducer, site, **kwargs):
    """image.reduceRegion over a site from site_region: the image is masked to the
    site's polygons and reduced over its bounding box, which is much faster than
    reducing over a complex polygon."""
    site_fc, bbox = site
    return image.clipToCollection(site_fc).reduceRegion(reducer, bbox, **kwargs)


def site_chunks(geom, chunk_km=INTACTNESS_CHUNK_KM):
    """A site cut on a grid into pieces at most chunk_km wide and tall (in
    GPD_CRS_METERS), as Earth Engine geometries; a site that fits in one piece is
    returned whole.

    The grid divides the site's bounding box evenly, and neighboring pieces share the
    same cut edges, so every part of the site is in exactly one piece.
    """
    to_m = Transformer.from_crs("EPSG:4326", GPD_CRS_METERS, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    (x0, x1), (y0, y1) = to_m.transform([min_lon, max_lon], [min_lat, max_lat])
    n_cols = max(1, math.ceil((x1 - x0) / (chunk_km * M_PER_KM)))
    n_rows = max(1, math.ceil((y1 - y0) / (chunk_km * M_PER_KM)))
    if n_cols * n_rows == 1:
        return [ee_geometry(geom)]

    # Grid lines in degrees, evenly spaced in meters; outer lines on the exact bounds
    xs, ys = np.linspace(x0, x1, n_cols + 1), np.linspace(y0, y1, n_rows + 1)
    lons = to_m.transform(xs, np.full_like(xs, y0), direction="INVERSE")[0]
    lats = to_m.transform(np.full_like(ys, x0), ys, direction="INVERSE")[1]
    lons[[0, -1]], lats[[0, -1]] = (min_lon, max_lon), (min_lat, max_lat)
    lon_lo, lat_lo = np.meshgrid(lons[:-1], lats[:-1])
    lon_hi, lat_hi = np.meshgrid(lons[1:], lats[1:])
    cells = shapely.box(lon_lo, lat_lo, lon_hi, lat_hi).ravel()

    chunks = []
    for piece in shapely.intersection(geom, cells):
        # Keep polygons only (cells that miss or just touch the site add empty
        # polygons, lines or points)
        polygons = [
            p
            for p in shapely.get_parts(piece)
            if p.geom_type == "Polygon" and not p.is_empty
        ]
        if polygons:
            chunks.append(ee_geometry(shapely.MultiPolygon(polygons)))
    return chunks
