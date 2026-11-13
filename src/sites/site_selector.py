"""
Read the active site group's cleaned sites (SITE_GEOMS, written by clean_sites.py), and
reduce Earth Engine images over a site.
"""

import ee
import geopandas as gpd
import pandas as pd

from utils.variables import SITE_GEOMS


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
