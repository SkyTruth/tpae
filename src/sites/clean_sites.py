"""
Select the active site group (SITE_GROUPS[SITE_GROUP]) from the latest WDPA feature
collection in Earth Engine (PAs only), clean their geometries, and save them to
SITE_GEOMS (GeoParquet in GCS), with a summary of the run in SITE_GEOMS_META (JSON).

A group is either a list of SITE_IDs or a list of (column, operator, value) filters,
all of which a site must pass. Filters on WDPA attributes (e.g. PRNT_ISO3) are
applied to each feature in Earth Engine, and a site is selected if any of its features
pass; filters on SITE_COLUMNS are applied to each cleaned site.

Every group uses only features in SITE_REALMS (Terrestrial and Coastal) that are not
Proposed, and leaves out sites designated too recently to analyze or with no
designation year. Sites whose features' GIS_AREA sums to less than MIN_GIS_AREA_KM2 are
left out before their geometries are downloaded; get_treatment_cells.py drops any
remaining site with too few land cells.

Cleaning keeps only polygonal parts, repairs invalid geometries, renames SITE_ID to
site_id, dissolves multi-part sites into one row per site, and simplifies boundaries by
SITE_SIMPLIFY_M.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import fsspec
import pandas as pd
from shapely import make_valid
from shapely.ops import unary_union

from cells.get_control_cells import init_ee
from utils.variables import (
    PROJECT,
    PAS_ASSET_ID,
    SITE_GROUP,
    SITE_GROUPS,
    SITE_GEOMS,
    SITE_GEOMS_META,
    SITE_SIMPLIFY_M,
    SITE_REALMS,
    ANALYSIS_START_YR,
    ANALYSIS_END_YR,
    GPD_CRS_METERS,
    CELL_SIZE_M,
    MIN_TREATMENT_CELLS,
)

M2_PER_KM2 = 1_000_000
# Smallest GIS_AREA sum (km2) of a site's features: 90% of the area MIN_TREATMENT_CELLS
# cells need, allowing for differences between WDPA's areas and the cleaned geometries
MIN_GIS_AREA_KM2 = 0.9 * MIN_TREATMENT_CELLS * CELL_SIZE_M**2 / M2_PER_KM2
# Site columns added by clean_sites; filters on these are applied to cleaned sites
SITE_COLUMNS = ("start_yr",)
EE_FILTERS = {
    "==": ee.Filter.eq,
    "!=": ee.Filter.neq,
    "<": ee.Filter.lt,
    "<=": ee.Filter.lte,
    ">": ee.Filter.gt,
    ">=": ee.Filter.gte,
    "in": lambda col, value: ee.Filter.inList(col, list(value)),
    "not in": lambda col, value: ee.Filter.inList(col, list(value)).Not(),
    "contains": ee.Filter.stringContains,
}
SITE_FILTERS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
    "contains": lambda s, v: s.str.contains(v, regex=False),
}


def is_filter_group(group):
    """True if group is a list of filters, False if it is a list of SITE_IDs."""
    return all(isinstance(item, tuple) for item in group)


def split_filters(filters):
    """Split filters into WDPA attribute filters and site column filters."""
    for col, op, _ in filters:
        if op not in EE_FILTERS:
            raise ValueError(f"Unknown operator {op!r} in filter on {col}")
    ee_filters = [f for f in filters if f[0] not in SITE_COLUMNS]
    site_filters = [f for f in filters if f[0] in SITE_COLUMNS]
    return ee_filters, site_filters


def wdpa_features():
    """WDPA polygons in SITE_REALMS that are not Proposed (server side)."""
    return (
        ee.FeatureCollection(PAS_ASSET_ID)
        .filter(ee.Filter.inList("REALM", SITE_REALMS))
        .filter(ee.Filter.neq("STATUS", "Proposed"))
    )


def select_site_ids(ee_filters):
    """SITE_IDs (server side) with a feature in wdpa_features passing every filter."""
    features = wdpa_features()
    for col, op, value in ee_filters:
        features = features.filter(EE_FILTERS[op](col, value))
    return features.aggregate_array("SITE_ID").distinct()


def site_gis_areas(site_ids):
    """One feature per site in site_ids (server side), with its SITE_ID and the sum of
    its wdpa_features' GIS_AREA (km2) as sum."""
    features = wdpa_features().filter(ee.Filter.inList("SITE_ID", site_ids))
    groups = features.reduceColumns(
        ee.Reducer.sum().group(groupField=1, groupName="SITE_ID"),
        ["GIS_AREA", "SITE_ID"],
    ).get("groups")
    return ee.FeatureCollection(ee.List(groups).map(lambda g: ee.Feature(None, g)))


def fetch_sites(site_ids):
    """Every polygon in wdpa_features of the given sites (GeoDataFrame)."""
    sites_fc = wdpa_features().filter(ee.Filter.inList("SITE_ID", site_ids))
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


def clean_sites(sites, simplify_m=SITE_SIMPLIFY_M):
    """Keep polygonal parts, rename SITE_ID to site_id, dissolve to one row per site,
    simplify boundaries, and add SITE_COLUMNS."""
    sites = sites.copy()
    sites["geometry"] = sites.geometry.apply(polygonal_part)
    sites = sites[sites.geometry.notna() & ~sites.geometry.is_empty]
    sites = sites.rename(columns={"SITE_ID": "site_id"})
    # Keep the first feature's attributes for multi-part sites
    sites = sites.dissolve(by="site_id", aggfunc="first", as_index=False)
    sites_m = sites.geometry.to_crs(GPD_CRS_METERS).simplify(simplify_m)
    sites["geometry"] = sites_m.to_crs(sites.crs)
    sites["start_yr"] = sites["STATUS_YR"].clip(lower=ANALYSIS_START_YR)
    return sites


def exclusion_reasons(sites, site_filters):
    """Why each site is left out of the group ("" if it is kept)."""
    reasons = pd.Series("", index=sites.index)
    rules = [
        (sites["STATUS_YR"] == 0, "no designation year (STATUS_YR = 0)"),
        (
            sites["start_yr"] > ANALYSIS_END_YR - 1,
            f"designated after {ANALYSIS_END_YR - 1}",
        ),
    ] + [
        (~SITE_FILTERS[op](sites[col], value), f"fails {col} {op} {value!r}")
        for col, op, value in site_filters
    ]
    for failed, reason in rules:
        reasons = reasons.where(~failed | (reasons != ""), reason)
    return reasons


def save_sites(
    group=SITE_GROUPS[SITE_GROUP], output_path=SITE_GEOMS, meta_path=SITE_GEOMS_META
):
    """Select, clean, and save the sites of a group, with a summary of the run."""
    init_ee(PROJECT)
    if is_filter_group(group):
        ee_filters, site_filters = split_filters(group)
        site_ids = select_site_ids(ee_filters)
    else:
        site_filters = []
        site_ids = list(group)

    # Leave out sites too small for MIN_TREATMENT_CELLS cells before downloading them
    areas = site_gis_areas(site_ids)
    small = areas.filter(ee.Filter.lt("sum", MIN_GIS_AREA_KM2))
    if is_filter_group(group):
        small_ids, n_small = [], small.size().getInfo()
    else:
        small_ids = small.aggregate_array("SITE_ID").getInfo()
        n_small = len(small_ids)
    features = fetch_sites(
        areas.filter(ee.Filter.gte("sum", MIN_GIS_AREA_KM2)).aggregate_array("SITE_ID")
    )
    sites = clean_sites(features)

    reasons = exclusion_reasons(sites, site_filters)
    excluded = sites.loc[reasons != "", "site_id"].astype(int)
    small_reason = (
        f"GIS_AREA below {MIN_GIS_AREA_KM2:g} km² (too small for "
        f"{MIN_TREATMENT_CELLS} treatment cells)"
    )
    n_left_out = {small_reason: n_small} if n_small else {}
    n_left_out |= {
        reason: int(n) for reason, n in reasons[reasons != ""].value_counts().items()
    }
    for reason, n in n_left_out.items():
        print(f"Left out {n} site(s): {reason}")
    if not is_filter_group(group):
        for site_id in small_ids:
            print(f"  {site_id}: {small_reason}")
        for site_id, reason in zip(excluded, reasons[reasons != ""]):
            print(f"  {site_id}: {reason}")
    sites = sites[reasons == ""].reset_index(drop=True)
    if sites.empty:
        raise RuntimeError(f"No sites left in group {SITE_GROUP}")

    missing = []
    if not is_filter_group(group):
        found = set(features["SITE_ID"].astype(int)) | set(small_ids)
        missing = sorted(set(group) - found)
        if missing:
            print(
                f"Warning: {len(missing)} SITE_ID(s) not found among {SITE_REALMS} "
                f"WDPA sites that are not Proposed: {missing}"
            )
    print(
        f"Fetched {len(features)} feature(s) for {len(sites) + len(excluded)} site(s); "
        f"kept {len(sites)}"
    )

    if "://" not in output_path:  # local path; gs:// needs no folder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(output_path)
    meta = {
        "site_group": SITE_GROUP,
        "group": [list(item) for item in group] if is_filter_group(group) else group,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wdpa_asset_id": PAS_ASSET_ID,
        "wdpa_update_time": ee.data.getAsset(PAS_ASSET_ID)["updateTime"],
        "n_features": len(features),
        "n_sites": len(sites),
        "n_left_out": n_left_out,
        "missing_site_ids": missing,
    }
    with fsspec.open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(sites)} site(s) to {output_path} and summary to {meta_path}")
    return sites


if __name__ == "__main__":
    save_sites()
