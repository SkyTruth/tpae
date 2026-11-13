"""
Read the active site group's cleaned sites (SITE_GEOMS, written by clean_sites.py),
reduce Earth Engine images over a site, and cut a site into pieces for intactness.
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


def site_region(geom):
    """A site for reduce_site: its polygons as an Earth Engine FeatureCollection, and
    its bounding box."""
    parts = getattr(geom, "geoms", [geom])
    site_fc = ee.FeatureCollection([ee.Feature(ee_geometry(p)) for p in parts])
    bbox = ee.Geometry.Rectangle(list(geom.bounds), proj="EPSG:4326", geodesic=False)
    return site_fc, bbox


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
