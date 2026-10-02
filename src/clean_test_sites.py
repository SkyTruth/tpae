"""
Extract a set of sites (TEST_SITE_IDS) from the WDPA and WDOECM feature collections in
Earth Engine, clean their geometries, and save them to TEST_SITES (GeoParquet).

Cleaning keeps only polygonal parts, repairs invalid geometries, renames SITE_ID to
WDPAID, and dissolves multi-part sites into one row per site.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

import ee
from shapely import make_valid
from shapely.ops import unary_union

from psm.get_control_cells import init_ee
from utils.variables import (
    PROJECT,
    PAS_ASSET_ID,
    OECMS_ASSET_ID,
    TEST_SITE_IDS,
    TEST_SITES,
)


def fetch_sites(site_ids):
    """Get all WDPA and WDOECM polygons for the given SITE_IDs as a GeoDataFrame."""
    pas = ee.FeatureCollection(PAS_ASSET_ID)
    oecms = ee.FeatureCollection(OECMS_ASSET_ID)
    sites_fc = (
        ee.FeatureCollection([pas, oecms])
        .flatten()
        .filter(ee.Filter.inList("SITE_ID", list(site_ids)))
    )
    # computeFeatures pages through results, so it is not limited to 5000 features
    return ee.data.computeFeatures(
        {"expression": sites_fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    ).set_crs("EPSG:4326")


def polygonal_part(geom):
    """Repair a geometry and keep only its polygon parts (drop stray lines/points)."""
    geom = make_valid(geom)
    parts = [
        g
        for g in getattr(geom, "geoms", [geom])
        if g.geom_type in ("Polygon", "MultiPolygon")
    ]
    return unary_union(parts) if parts else None


def clean_sites(sites):
    """Keep polygonal parts, rename SITE_ID to WDPAID, and dissolve to one row per site."""
    sites = sites.copy()
    sites["geometry"] = sites.geometry.apply(polygonal_part)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.rename(columns={"SITE_ID": "WDPAID"})
    # Keep the first feature's attributes for multi-part sites
    return sites.dissolve(by="WDPAID", aggfunc="first", as_index=False)


def save_sites(site_ids=TEST_SITE_IDS, output_path=TEST_SITES):
    """Extract, clean, and save the given sites."""
    init_ee(PROJECT)
    raw = fetch_sites(site_ids)
    sites = clean_sites(raw)

    missing = sorted(set(site_ids) - set(sites["WDPAID"]))
    if missing:
        print(f"Warning: {len(missing)} SITE_ID(s) not found in WDPA/WDOECM: {missing}")
    print(
        f"Fetched {len(raw)} feature(s) for {len(sites)} site(s); "
        f"realms: {sites['REALM'].value_counts().to_dict()}"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(output_path)
    print(f"Saved {len(sites)} site(s) to {output_path}")
    return sites


if __name__ == "__main__":
    save_sites()
