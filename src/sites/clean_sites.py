"""
Clean the sites of the active site group (SITE_GROUP) and save them to SITE_GEOMS
(GeoParquet in GCS), with a summary of the run in SITE_GEOMS_META (JSON).

Sites are chosen from the eligible sites in SITE_CHUNKS, written by chunker.py. A group
named "chunk_<n>" is chunk n; any other group is SITE_GROUPS[SITE_GROUP], either a list
of SITE_IDs or a list of (column, operator, value) filters on SITE_CHUNKS columns, all of
which a site must pass. get_treatment_cells.py drops any site with too few land cells.

Each site's features are fetched with chunker.wdpa_features, the same features
chunker.py counted. Cleaning keeps only polygonal parts, repairs invalid geometries,
renames SITE_ID to site_id, dissolves multi-part sites into one row per site, and
simplifies boundaries by SITE_SIMPLIFY_M. Sites no longer in WDPA, or with no polygon
parts, are recorded as dropped in DROPPED_SITES, which clears every later step's records.
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
from sites.chunker import wdpa_features
from sites.dropped_sites import record_dropped
from utils.retries import with_retries
from utils.variables import (
    PROJECT,
    PAS_ASSET_ID,
    SITE_GROUP,
    SITE_GROUPS,
    SITE_CHUNKS,
    SITE_GEOMS,
    SITE_GEOMS_META,
    DROPPED_SITES,
    SITE_SIMPLIFY_M,
    ANALYSIS_START_YR,
    GPD_CRS_METERS,
)

# Groups named CHUNK_GROUP_PREFIX + n are chunk n of SITE_CHUNKS
CHUNK_GROUP_PREFIX = "chunk_"
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


def select_sites(chunks, group_name=SITE_GROUP):
    """Rows of chunks (the SITE_CHUNKS table) in a group, and the group's listed
    SITE_IDs that are not in chunks (not eligible)."""
    if group_name.startswith(CHUNK_GROUP_PREFIX):
        chunk = int(group_name.removeprefix(CHUNK_GROUP_PREFIX))
        return chunks[chunks["chunk"] == chunk], []
    group = SITE_GROUPS[group_name]
    if not is_filter_group(group):
        not_eligible = sorted(set(group) - set(chunks["site_id"]))
        return chunks[chunks["site_id"].isin(group)], not_eligible
    keep = pd.Series(True, index=chunks.index)
    for col, op, value in group:
        if op not in SITE_FILTERS:
            raise ValueError(f"Unknown operator {op!r} in filter on {col}")
        keep &= SITE_FILTERS[op](chunks[col], value)
    return chunks[keep], []


def fetch_sites(site_ids):
    """Every polygon in wdpa_features of the given sites (GeoDataFrame)."""
    sites_fc = wdpa_features().filter(ee.Filter.inList("SITE_ID", site_ids))
    # computeFeatures pages through results, so it is not limited to 5000 features
    return with_retries(
        ee.data.computeFeatures,
        {"expression": sites_fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"},
        label="WDPA features",
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
    simplify boundaries, and add start_yr."""
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


def save_sites(
    group_name=SITE_GROUP,
    output_path=SITE_GEOMS,
    meta_path=SITE_GEOMS_META,
    dropped_path=DROPPED_SITES,
):
    """Clean and save the sites of a group, with a summary of the run, and record the
    sites dropped."""
    init_ee(PROJECT)
    chunks = pd.read_parquet(SITE_CHUNKS)
    group_sites, not_eligible = select_sites(chunks, group_name)
    if not_eligible:
        print(
            f"Left out {len(not_eligible)} SITE_ID(s) not among the eligible sites in "
            f"{SITE_CHUNKS}: {not_eligible}"
        )
    if group_sites.empty:
        raise RuntimeError(f"No eligible sites in group {group_name}")

    wdpa = with_retries(ee.data.getAsset, PAS_ASSET_ID, label="WDPA asset")
    wdpa_update_time = wdpa["updateTime"]
    chunks_wdpa_update_time = chunks["wdpa_update_time"].iloc[0]
    if wdpa_update_time != chunks_wdpa_update_time:
        print(
            f"Warning: WDPA was updated at {wdpa_update_time}, but {SITE_CHUNKS} was "
            f"built from the release updated at {chunks_wdpa_update_time}"
        )

    site_ids = [int(site_id) for site_id in group_sites["site_id"]]
    features = fetch_sites(site_ids)
    if features.empty:
        raise RuntimeError(f"No WDPA features found for group {group_name}")
    sites = clean_sites(features)
    if sites.empty:
        raise RuntimeError(f"No sites left in group {group_name}")

    missing = sorted(set(site_ids) - set(features["SITE_ID"].astype(int)))
    if missing:
        print(
            f"Warning: {len(missing)} SITE_ID(s) no longer found in WDPA by "
            f"wdpa_features: {missing}"
        )
    print(
        f"Fetched {len(features)} feature(s) for {len(site_ids)} site(s); "
        f"kept {len(sites)}"
    )

    if "://" not in output_path:  # local path; gs:// needs no folder
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(output_path)
    group = SITE_GROUPS.get(group_name)
    if group is not None and is_filter_group(group):
        group = [list(item) for item in group]
    meta = {
        "site_group": group_name,
        "group": group,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wdpa_asset_id": PAS_ASSET_ID,
        "wdpa_update_time": wdpa_update_time,
        "site_chunks_wdpa_update_time": chunks_wdpa_update_time,
        "n_features": len(features),
        "n_sites": len(sites),
        "not_eligible_site_ids": not_eligible,
        "missing_site_ids": missing,
    }
    with fsspec.open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(sites)} site(s) to {output_path} and summary to {meta_path}")

    no_parts = set(features["SITE_ID"].astype(int)) - set(sites["site_id"].astype(int))
    dropped = {site_id: "no longer in WDPA" for site_id in missing}
    dropped |= {site_id: "no polygon parts" for site_id in sorted(no_parts)}
    record_dropped("clean_sites", dropped, path=dropped_path)
    return sites


if __name__ == "__main__":
    save_sites()
