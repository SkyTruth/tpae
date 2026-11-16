"""
Extract covariate values for every candidate cell of each site in the active SITE_GROUP
and save them to GCS, one parquet file per site.

Run it once per site group, after export_covariate_stack.py, get_treatment_cells.py
and get_control_cells.py; MDM_match_cells.py then reads the saved files, so matching
can be re-run quickly. Every cell is one pixel of the exported 1 km covariate stack, so
its covariates are read as pixel values, with country, ecoregion and biome painted onto
the same grid.

Cells from all sites are pooled, so a cell shared by several sites is extracted once,
and read from Earth Engine in tiles of the stack's grid, several at a time. A site is
saved once all of its cells are extracted.

A site is skipped if its file is newer than the treatment and control cell files and the
covariate stack, so re-running picks up where a stalled or failed run left off, and regenerated cells are
re-extracted automatically. Use --overwrite to re-extract every site.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

cur = Path.cwd().resolve()
for parent in [cur] + list(cur.parents):
    if parent.name == "tpae":
        os.chdir(parent)
        break

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

import ee
import gcsfs
import geopandas as gpd
import pandas as pd

from utils.variables import (
    PROJECT,
    SITE_GROUP,
    GCS_BUCKET,
    CELL_COVARIATES_PREFIX,
    TREATMENT_CELLS,
    CONTROL_CELLS,
    EE_DEADLINE_MS,
    EE_N_WORKERS,
)


def cell_covariates_path(site_id):
    """GCS path of a site's extracted cell covariates."""
    return f"gs://{GCS_BUCKET}/{CELL_COVARIATES_PREFIX}{site_id}.parquet"


def is_up_to_date(fs, path, inputs_modified):
    """True if the file exists and is newer than the cell files it was extracted from."""
    return fs.exists(path) and fs.modified(path) > inputs_modified


def extract_site_covariates(site_ids=None, overwrite=False, n_workers=EE_N_WORKERS):
    """Extract and save each site's cell covariates; return the sites that failed.

    site_ids: sites to extract (default: every site in SITE_GEOMS).
    """
    # Imported here so the module can be imported (e.g. for cell_covariates_path)
    # without loading the Earth Engine pipeline
    from cells.load_all_cells import build_all_site_cells
    from sites.site_selector import load_site_ids
    from utils.retries import report_failures, with_retries
    from covariates.export_covariate_stack import (
        covariate_stack_updated,
        load_covariate_stack,
    )
    from covariates.cell_features import (
        TILE_PX,
        covariate_image,
        drop_invalid_cells,
        extract_covariates,
        stack_pixels,
    )

    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    ee.data.setDeadline(EE_DEADLINE_MS)

    if site_ids is None:
        site_ids = load_site_ids()
    image, country_names = covariate_image(load_covariate_stack())

    fs = gcsfs.GCSFileSystem()
    inputs_modified = max(
        fs.modified(TREATMENT_CELLS),
        fs.modified(CONTROL_CELLS),
        covariate_stack_updated(),
    )
    todo = [
        str(site_id)
        for site_id in site_ids
        if overwrite
        or not is_up_to_date(fs, cell_covariates_path(site_id), inputs_modified)
    ]
    if len(todo) < len(site_ids):
        print(f"Up to date, skipping {len(site_ids) - len(todo)} site(s)")

    cells = build_all_site_cells(
        todo, gpd.read_parquet(TREATMENT_CELLS), gpd.read_parquet(CONTROL_CELLS)
    )
    # Sites with no cells are saved as empty tables, so they count as done
    for site_id in sorted(set(todo) - set(cells["site_id"])):
        pd.DataFrame().to_parquet(cell_covariates_path(site_id), index=False)
        print(f"Site {site_id}: no cells, saved empty table")

    # Each unique cell is extracted once; a chunk is the cells in one TILE_PX x TILE_PX
    # tile of stack pixels, numbered in the order sites first use them
    site_order = cells["site_id"].map({s: i for i, s in enumerate(todo)})
    unique_cells = (
        cells.assign(_order=site_order)
        .sort_values(["_order", "cell_ID"])
        .drop_duplicates("cell_key")[["cell_key", "site_id", "geometry"]]
    )
    unique_cells["ix"], unique_cells["iy"] = stack_pixels(unique_cells)
    unique_cells["chunk"] = unique_cells.groupby(
        [unique_cells["ix"] // TILE_PX, unique_cells["iy"] // TILE_PX], sort=False
    ).ngroup()
    chunk_cells = {
        c: g[["cell_key", "ix", "iy"]] for c, g in unique_cells.groupby("chunk")
    }
    cells["chunk"] = cells["cell_key"].map(unique_cells.set_index("cell_key")["chunk"])
    n_chunks = len(chunk_cells)
    site_rows = cells.groupby("site_id").indices
    site_chunks = {s: set(cells["chunk"].iloc[i]) for s, i in site_rows.items()}
    chunk_sites = cells.groupby("chunk")["site_id"].unique()
    print(
        f"Extracting {len(unique_cells)} unique cells for {len(site_rows)} sites "
        f"({len(cells)} site cells) in {n_chunks} tiles, {n_workers} at a time"
    )

    def extract_chunk(chunk):
        return with_retries(
            extract_covariates,
            chunk_cells[chunk],
            image,
            country_names,
            label=f"Chunk {chunk}",
        )

    def save_site(site_id):
        rows = cells.iloc[site_rows[site_id]]
        results = pd.concat([chunk_results[c] for c in sorted(site_chunks[site_id])])
        cells_df = rows[["cell_ID", "protected", "cell_key"]].merge(
            results, on="cell_key"
        )
        print(f"\nSite {site_id}: {len(rows)} cells")
        n_unjoined = len(rows) - len(cells_df)
        if n_unjoined > 0:
            print(
                f"Dropped {n_unjoined} cell(s) with no matching country or ecoregion."
            )
        cells_df = cells_df.drop(columns="cell_key").sort_values("cell_ID")
        cells_df = drop_invalid_cells(cells_df.reset_index(drop=True))
        # Columns in alphabetical order, as Earth Engine returns them
        cells_df = cells_df[sorted(cells_df.columns)]
        path = cell_covariates_path(site_id)
        cells_df.to_parquet(path, index=False)
        print(f"Saved {len(cells_df)} cells to {path}")

    chunk_results = {}
    failed_chunks = set()
    remaining = {s: set(chunks) for s, chunks in site_chunks.items()}
    # Sites not yet saved for each chunk, so a chunk's results are freed once unused
    chunk_users = {c: len(sites) for c, sites in chunk_sites.items()}
    failed = {}
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = {executor.submit(extract_chunk, c): c for c in range(n_chunks)}
        for n_done, future in enumerate(as_completed(futures), start=1):
            chunk = futures[future]
            try:
                chunk_results[chunk] = future.result()
                print(f"Chunk {chunk} done [{n_done}/{n_chunks}]")
            except Exception as exc:
                print(f"FAILED: chunk {chunk}: {exc}")
                failed_chunks.add(chunk)
            # Save each site whose chunks are now all done
            for site_id in chunk_sites[chunk]:
                remaining[site_id].discard(chunk)
                if remaining[site_id]:
                    continue
                try:
                    if site_chunks[site_id] & failed_chunks:
                        raise RuntimeError("a chunk with its cells failed")
                    save_site(site_id)
                except Exception as exc:
                    print(f"FAILED: site {site_id}: {exc}")
                    failed[site_id] = str(exc)
                for c in site_chunks[site_id]:
                    chunk_users[c] -= 1
                    if chunk_users[c] == 0:
                        chunk_results.pop(c, None)
    finally:
        # Stop unstarted chunks if interrupted
        executor.shutdown(cancel_futures=True)

    failed = {s: failed[str(s)] for s in site_ids if str(s) in failed}
    print(f"\nSite group {SITE_GROUP}: {len(site_ids) - len(failed)} sites done")
    report_failures(failed, len(site_ids))
    if failed:
        print("Re-run to retry them")
    return list(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract every site"
    )
    extract_site_covariates(overwrite=parser.parse_args().overwrite)
