"""
Read the active site group's cleaned sites (SITE_GEOMS, written by clean_sites.py).
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
