"""
Split every eligible WDPA site into chunks of at most CHUNK_SIZE sites, and save each
site's chunk to SITE_CHUNKS (Parquet in GCS) and the index of the last chunk to
CHUNKS_META (JSON). Chunks are numbered from 0.

Eligible sites have features in wdpa_features (Terrestrial and Coastal PAs that are not
Proposed, designated from 1 to ANALYSIS_END_YR - 1) whose GIS_AREA sums to at least
MIN_GIS_AREA_KM2. get_treatment_cells.py drops any remaining site with too few land cells.

Sites are sorted by country or territory (ISO3), then site_id. A chunk ends early at a
country boundary if it still has at least MIN_CHUNK_SIZE sites.

Run once, before clean_sites.py; re-running after a WDPA update moves chunk boundaries.
"""

from pathlib import Path
import json
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import fsspec

from cells.get_control_cells import init_ee
from utils.retries import with_retries
from utils.variables import (
    PROJECT,
    PAS_ASSET_ID,
    SITE_REALMS,
    ANALYSIS_END_YR,
    CELL_SIZE_M,
    MIN_TREATMENT_CELLS,
    SITE_CHUNKS,
    CHUNKS_META,
)

M2_PER_KM2 = 1_000_000
# Smallest GIS_AREA sum (km2) of a site's features: 90% of the area MIN_TREATMENT_CELLS
# cells need, allowing for differences between WDPA's areas and the cleaned geometries
MIN_GIS_AREA_KM2 = 0.9 * MIN_TREATMENT_CELLS * CELL_SIZE_M**2 / M2_PER_KM2
# Most sites in a chunk
CHUNK_SIZE = 700
# Fewest sites in a chunk that ends early at a country boundary
MIN_CHUNK_SIZE = round(0.85 * CHUNK_SIZE)
# WDPA attributes downloaded for each feature
WDPA_COLUMNS = ["SITE_ID", "GIS_AREA", "ISO3"]


def wdpa_features():
    """WDPA polygons in SITE_REALMS that are not Proposed, designated from 1 to
    ANALYSIS_END_YR - 1 (STATUS_YR 0 means unknown) (server side)."""
    return (
        ee.FeatureCollection(PAS_ASSET_ID)
        .filter(ee.Filter.inList("REALM", SITE_REALMS))
        .filter(ee.Filter.neq("STATUS", "Proposed"))
        .filter(ee.Filter.gte("STATUS_YR", 1))
        .filter(ee.Filter.lte("STATUS_YR", ANALYSIS_END_YR - 1))
    )


def site_areas():
    """One row per site with features in wdpa_features: site_id, ISO3, and
    gis_area_km2 (the sum of its features' GIS_AREA)."""
    features = wdpa_features().select(WDPA_COLUMNS, None, False)
    # computeFeatures pages through results, so it is not limited to 5000 features
    features = with_retries(
        ee.data.computeFeatures,
        {"expression": features, "fileFormat": "PANDAS_DATAFRAME"},
        label="WDPA features",
    )[WDPA_COLUMNS]
    sites = (
        features.groupby("SITE_ID", as_index=False)
        # Alphabetically first ISO3 of a site's features
        .agg(ISO3=("ISO3", "min"), gis_area_km2=("GIS_AREA", "sum"))
        .rename(columns={"SITE_ID": "site_id"})
        .astype({"site_id": "int64"})
    )
    print(f"Downloaded {len(features)} feature(s) of {len(sites)} site(s)")
    return sites


def assign_chunks(countries, chunk_size=CHUNK_SIZE, min_chunk_size=MIN_CHUNK_SIZE):
    """Chunk of each site, from each site's country in sorted order. A chunk takes up to
    chunk_size sites, ending at its last country boundary that leaves at least
    min_chunk_size sites, if there is one."""
    countries = list(countries)
    chunks, start, chunk = [], 0, 0
    while start < len(countries):
        end = min(start + chunk_size, len(countries))
        if end < len(countries):
            for cut in range(end, start + min_chunk_size - 1, -1):
                if countries[cut - 1] != countries[cut]:
                    end = cut
                    break
        chunks += [chunk] * (end - start)
        start, chunk = end, chunk + 1
    return chunks


def save_chunks(chunks_path=SITE_CHUNKS, meta_path=CHUNKS_META):
    """Assign every eligible site to a chunk, and save the table and the last chunk
    index."""
    init_ee(PROJECT)
    wdpa = with_retries(ee.data.getAsset, PAS_ASSET_ID, label="WDPA asset")
    wdpa_update_time = wdpa["updateTime"]
    sites = site_areas()
    small = sites["gis_area_km2"] < MIN_GIS_AREA_KM2
    print(f"Left out {small.sum()} site(s): GIS_AREA below {MIN_GIS_AREA_KM2:g} km²")
    sites = sites[~small].sort_values(["ISO3", "site_id"]).reset_index(drop=True)
    sites["chunk"] = assign_chunks(sites["ISO3"])
    sites["wdpa_update_time"] = wdpa_update_time
    sites = sites[["site_id", "chunk", "ISO3", "gis_area_km2", "wdpa_update_time"]]

    sites.to_parquet(chunks_path, index=False)
    last_chunk_index = int(sites["chunk"].max())
    with fsspec.open(meta_path, "w") as f:
        json.dump({"last_chunk_index": last_chunk_index}, f, indent=2)

    for chunk, chunk_sites in sites.groupby("chunk"):
        countries = chunk_sites["ISO3"].unique()
        print(
            f"Chunk {chunk}: {len(chunk_sites)} sites, {len(countries)} countries "
            f"({countries[0]}–{countries[-1]}), largest site "
            f"{chunk_sites['gis_area_km2'].max():,.0f} km²"
        )
    print(
        f"Saved {len(sites)} site(s) in chunks 0–{last_chunk_index} to {chunks_path}, "
        f"and the last chunk index to {meta_path}"
    )
    return sites


if __name__ == "__main__":
    save_chunks()
