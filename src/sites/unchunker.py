"""
Merge the outputs of every chunk (0 to last_chunk_index in CHUNKS_META) into global
files in GLOBAL_RESULTS_PREFIX, each table with a chunk column:
- relative and absolute results, report card, and match diagnostics, with one row per
  kept site (cleaned and not dropped at any step);
- dropped_sites.csv: every dropped site, with the step that dropped it and why (reason
  "failed: <error>" for sites that failed with an error), one row per step for sites
  dropped at both steps of a rung (e.g. relative and absolute effectiveness);
- site_geoms.parquet: every cleaned site's geometry;
- summary.json: chunks merged and site counts.

Before writing, checks that every chunk:
- is complete: it has every file, every site was cleaned or dropped at clean_sites, and
  every site not dropped has a row in each of RESULT_TABLES, without errors (an error
  not recorded as a drop means a step was interrupted, or an earlier step re-ran), and
  a scored-pairs file from relative_effectiveness.py;
- holds only its own sites, each once (in dropped_sites, once per step);
- has each file newer than the files it was built from (INPUT_FILES);
- was cleaned from the chunk table's WDPA release.
Nothing is written if a check fails. With --allow-partial, incomplete chunks are merged
anyway, leaving out sites without results, so the global files can be checked while the
run is in progress.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import fsspec
import geopandas as gpd
import pandas as pd

from matching.match_cells import scored_pairs_path
from sites.clean_sites import CHUNK_GROUP_PREFIX
from sites.dropped_sites import COLUMNS as DROPPED_COLUMNS, FAILED_PREFIX
from utils.variables import (
    GCS_BUCKET,
    MATCHES_PREFIX_TEMPLATE,
    SITE_CHUNKS,
    CHUNKS_META,
    SITE_GEOMS_TEMPLATE,
    SITE_GEOMS_META_TEMPLATE,
    DROPPED_SITES_TEMPLATE,
    TREATMENT_CELLS_TEMPLATE,
    CONTROL_CELLS_TEMPLATE,
    MATCH_DIAGNOSTICS_TEMPLATE,
    MATCH_REPORT_CARD_TEMPLATE,
    RELATIVE_RESULTS_TEMPLATE,
    ABSOLUTE_RESULTS_TEMPLATE,
    GLOBAL_RESULTS_PREFIX,
)

# Each chunk's files, by name
CHUNK_FILES = {
    "site_geoms": SITE_GEOMS_TEMPLATE,
    "site_geoms_meta": SITE_GEOMS_META_TEMPLATE,
    "dropped_sites": DROPPED_SITES_TEMPLATE,
    "treatment_cells": TREATMENT_CELLS_TEMPLATE,
    "control_cells": CONTROL_CELLS_TEMPLATE,
    "diagnostics": MATCH_DIAGNOSTICS_TEMPLATE,
    "report_card": MATCH_REPORT_CARD_TEMPLATE,
    "relative_results": RELATIVE_RESULTS_TEMPLATE,
    "absolute_results": ABSOLUTE_RESULTS_TEMPLATE,
}
# Files that must be newer than the files they are built from
INPUT_FILES = {
    "treatment_cells": ["site_geoms"],
    "control_cells": ["treatment_cells"],
    "diagnostics": ["control_cells"],
    "report_card": ["diagnostics"],
    "relative_results": ["report_card"],
    "absolute_results": ["site_geoms"],
}
# Tables with one row per site, merged for kept sites into GLOBAL_RESULTS_PREFIX
RESULT_TABLES = ["relative_results", "absolute_results", "report_card", "diagnostics"]
# Files checked for their last modified time at once
N_WORKERS = 16


def chunk_paths(chunk, chunk_files=CHUNK_FILES):
    """{file name: path} of a chunk's files."""
    group = f"{CHUNK_GROUP_PREFIX}{chunk}"
    return {
        name: template.format(group=group) for name, template in chunk_files.items()
    }


def modified_time(path):
    """A file's last modified time, or None if it doesn't exist."""
    fs, fs_path = fsspec.core.url_to_fs(path)
    try:
        return pd.Timestamp(fs.modified(fs_path))
    except FileNotFoundError:
        return None


def read_json(path):
    """Contents of a JSON file."""
    with fsspec.open(path) as f:
        return json.load(f)


def stale_files(files, times):
    """Problems with files (name: path) older than a file they are built from."""
    return [
        f"{name} is older than {input_name}"
        for name, input_names in INPUT_FILES.items()
        for input_name in input_names
        if times[files[name]] is not None
        and times[files[input_name]] is not None
        and times[files[name]] < times[files[input_name]]
    ]


def wdpa_problems(meta, wdpa_update_time):
    """Problems with a chunk's site_geoms summary not matching the chunk table's WDPA
    release."""
    return [
        f"{key} in site_geoms_meta is {meta.get(key)}, not {wdpa_update_time}"
        for key in ("wdpa_update_time", "site_chunks_wdpa_update_time")
        if meta.get(key) != wdpa_update_time
    ]


def read_tables(files, times):
    """A chunk's site geometries, dropped sites and RESULT_TABLES, for the files that
    exist."""
    tables = {}
    if times[files["site_geoms"]] is not None:
        tables["site_geoms"] = gpd.read_parquet(files["site_geoms"])
    for name in ["dropped_sites", *RESULT_TABLES]:
        if times[files[name]] is not None:
            tables[name] = pd.read_csv(files[name])
    for table in tables.values():
        table["site_id"] = table["site_id"].astype("int64")
    return tables


def membership_problems(tables, chunk_sites):
    """Problems with tables holding sites of other chunks, or a site more than once (in
    dropped_sites, more than once at a step)."""
    problems = []
    for name, table in tables.items():
        outside = sorted(set(table["site_id"]) - chunk_sites)
        if outside:
            problems.append(
                f"{name} has {len(outside)} site(s) not in this chunk: {outside[:10]}"
            )
        keys = ["site_id", "step"] if name == "dropped_sites" else ["site_id"]
        if table.duplicated(keys).any():
            problems.append(f"{name} has a site more than once")
    return problems


def sort_sites(tables, chunk_sites):
    """A chunk's kept site_ids, its dropped sites (site_id, step, reason), and what is
    missing for the rest (sites neither cleaned nor dropped, rows in RESULT_TABLES, or
    rows with errors).

    tables: the chunk's tables, with site_geoms.
    """
    dropped = tables.get("dropped_sites", pd.DataFrame(columns=DROPPED_COLUMNS))
    dropped = dropped[DROPPED_COLUMNS].astype({"site_id": "int64"})
    cleaned = set(tables["site_geoms"]["site_id"])
    missing = []
    n_unrecorded = len(chunk_sites - cleaned - set(dropped["site_id"]))
    if n_unrecorded:
        missing.append(
            f"{n_unrecorded} site(s) neither cleaned nor dropped (re-run clean_sites)"
        )

    to_analyze = cleaned - set(dropped["site_id"])
    kept = set(to_analyze)
    for name in RESULT_TABLES:
        if name not in tables:
            kept = set()
            continue
        table = tables[name]
        has_row = ok = set(table["site_id"])
        if "error" in table:
            ok = set(table.loc[table["error"].isna(), "site_id"])
        n_missing = len(to_analyze - has_row)
        n_failed = len((to_analyze & has_row) - ok)
        if n_missing:
            missing.append(f"{n_missing} site(s) missing from {name}")
        if n_failed:
            missing.append(f"{n_failed} site(s) with errors in {name}")
        kept &= ok
    return kept, dropped, missing


def sites_without_scored_pairs(site_ids, group):
    """site_ids of a group with no scored-pairs file (scored_pairs_path), from one
    listing of the group's matches folder."""
    folder = f"gs://{GCS_BUCKET}/{MATCHES_PREFIX_TEMPLATE.format(group=group)}"
    fs, fs_folder = fsspec.core.url_to_fs(folder)
    try:
        names = {path.rsplit("/", 1)[-1] for path in fs.ls(fs_folder, detail=False)}
    except FileNotFoundError:
        names = set()
    return {s for s in site_ids if scored_pairs_path(s).rsplit("/", 1)[-1] not in names}


def merge(frames):
    """Tables stacked, or an empty table with site_id and chunk if there are none."""
    if not frames:
        return pd.DataFrame(
            {"site_id": pd.Series(dtype="int64"), "chunk": pd.Series(dtype="int64")}
        )
    return pd.concat(frames, ignore_index=True)


def unchunk(
    allow_partial=False,
    chunks_path=SITE_CHUNKS,
    meta_path=CHUNKS_META,
    output_prefix=GLOBAL_RESULTS_PREFIX,
    chunk_files=CHUNK_FILES,
):
    """Check every chunk's outputs and merge them into output_prefix."""
    chunks = pd.read_parquet(chunks_path)
    last_chunk_index = read_json(meta_path)["last_chunk_index"]
    wdpa_update_time = chunks["wdpa_update_time"].iloc[0]
    files = {c: chunk_paths(c, chunk_files) for c in range(last_chunk_index + 1)}
    all_paths = [path for paths in files.values() for path in paths.values()]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        times = dict(zip(all_paths, executor.map(modified_time, all_paths)))

    errors, incomplete = [], {}
    frames = {name: [] for name in ["site_geoms", "dropped_sites", *RESULT_TABLES]}
    for chunk, paths in files.items():
        chunk_sites = set(chunks.loc[chunks["chunk"] == chunk, "site_id"])
        tables = read_tables(paths, times)
        meta = None
        if times[paths["site_geoms_meta"]] is not None:
            meta = read_json(paths["site_geoms_meta"])
        problems = stale_files(paths, times) + membership_problems(tables, chunk_sites)
        if meta is not None:
            problems += wdpa_problems(meta, wdpa_update_time)
        errors += [f"Chunk {chunk}: {problem}" for problem in problems]

        missing = [f"no {name}" for name, path in paths.items() if times[path] is None]
        if "site_geoms" in tables:
            kept, dropped, unanalyzed = sort_sites(tables, chunk_sites)
            no_pairs = sites_without_scored_pairs(kept, f"{CHUNK_GROUP_PREFIX}{chunk}")
            if no_pairs:
                unanalyzed.append(f"{len(no_pairs)} site(s) without scored pairs")
                kept -= no_pairs
            missing += unanalyzed
            chunk_frames = {
                "site_geoms": tables["site_geoms"],
                "dropped_sites": dropped,
                **{
                    name: tables[name][tables[name]["site_id"].isin(kept)].copy()
                    for name in RESULT_TABLES
                    if name in tables
                },
            }
            for name, table in chunk_frames.items():
                table.insert(1, "chunk", chunk)
                frames[name].append(table)
        if missing:
            incomplete[chunk] = missing
        print(f"Chunk {chunk}: " + ("; ".join(missing) if missing else "complete"))

    if incomplete and not allow_partial:
        errors.append(
            f"{len(incomplete)} incomplete chunk(s): {sorted(incomplete)}; use "
            "--allow-partial to merge them anyway"
        )
    if errors:
        print("\n".join(errors))
        raise RuntimeError(
            f"{len(errors)} check(s) failed; nothing written to {output_prefix}"
        )

    merged = {name: merge(frames[name]) for name in RESULT_TABLES}
    dropped = merge(frames["dropped_sites"]).merge(
        chunks[["site_id", "ISO3"]], on="site_id", how="left"
    )
    dropped = dropped.reindex(columns=["site_id", "chunk", "ISO3", "step", "reason"])
    if "://" not in output_prefix:  # local path; gs:// needs no folder
        Path(output_prefix).mkdir(parents=True, exist_ok=True)
    for name, table in merged.items():
        table.to_csv(f"{output_prefix}{name}.csv", index=False)
    dropped.to_csv(f"{output_prefix}dropped_sites.csv", index=False)
    if frames["site_geoms"]:
        site_geoms = gpd.GeoDataFrame(
            merge(frames["site_geoms"]), crs=frames["site_geoms"][0].crs
        )
        site_geoms.to_parquet(f"{output_prefix}site_geoms.parquet")

    n_kept = len(merged["relative_results"])
    failed = dropped[dropped["reason"].astype(str).str.startswith(FAILED_PREFIX)]
    # Sites, not rows: a site can be dropped at both steps of a rung
    n_dropped = dropped["site_id"].nunique()
    n_failed = failed["site_id"].nunique()
    report_card = merged["report_card"]
    tiers = report_card["assessable"] if "assessable" in report_card else pd.Series()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_chunk_index": last_chunk_index,
        "complete_chunks": [c for c in files if c not in incomplete],
        "incomplete_chunks": {str(c): parts for c, parts in incomplete.items()},
        "wdpa_update_time": wdpa_update_time,
        "n_eligible_sites": len(chunks),
        "n_kept_sites": n_kept,
        "n_dropped_sites": {
            k: int(n) for k, n in dropped["step"].value_counts().items()
        },
        "n_failed_sites": {k: int(n) for k, n in failed["step"].value_counts().items()},
        # In steps that have not run yet, or were interrupted
        "n_not_merged_sites": len(chunks) - n_kept - n_dropped,
        "assessable_counts": {k: int(n) for k, n in tiers.value_counts().items()},
    }
    with fsspec.open(f"{output_prefix}summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n{n_kept} kept, {n_dropped} dropped ({n_failed} failed) and "
        f"{summary['n_not_merged_sites']} not merged of {len(chunks)} sites"
    )
    print(f"Report card tiers of kept sites: {summary['assessable_counts']}")
    print(f"Saved merged tables, dropped sites and summary to {output_prefix}")
    return merged, dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="merge incomplete chunks instead of stopping",
    )
    unchunk(allow_partial=parser.parse_args().allow_partial)
