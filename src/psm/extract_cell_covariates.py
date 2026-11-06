"""
Extract covariate values for every candidate cell of each site in the active SITE_GROUP
and save them to GCS, one parquet file per site.

This is the slow Earth Engine step. Run it once per site group, after
get_treatment_cells.py and get_control_cells.py; mahalanobis_experiment.py then reads
the saved files, so matching can be re-run quickly.

A site is skipped if its file is newer than both the treatment and control cell files,
so re-running picks up where a stalled or failed run left off, and regenerated cells are
re-extracted automatically. Use --overwrite to re-extract every site.
"""

import argparse
import os
import sys
from pathlib import Path

cur = Path.cwd().resolve()
for parent in [cur] + list(cur.parents):
    if parent.name == "tpae":
        os.chdir(parent)
        break

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

import ee
import gcsfs

from utils.variables import (
    PROJECT,
    EE_CRS_METERS,
    PSM_CELL_SIZE,
    TEST_SITE_IDS,
    SITE_GROUP,
    GCS_BUCKET,
    CELL_COVARIATES_PREFIX,
    TREATMENT_CELLS,
    CONTROL_CELLS,
    EE_DEADLINE_MS,
)


def cell_covariates_path(site_id):
    """GCS path of a site's extracted cell covariates."""
    return f"gs://{GCS_BUCKET}/{CELL_COVARIATES_PREFIX}{site_id}.parquet"


def is_up_to_date(fs, path, inputs_modified):
    """True if the file exists and is newer than the cell files it was extracted from."""
    return fs.exists(path) and fs.modified(path) > inputs_modified


def extract_site_covariates(site_ids=TEST_SITE_IDS, overwrite=False):
    """Extract and save each site's cell covariates; return the sites that failed."""
    # Imported here so the module can be imported (e.g. for cell_covariates_path)
    # without loading the Earth Engine pipeline
    from absolute_effectiveness.site_selector import SiteSelector
    from psm.prepare_pa_grid import load_pa_candidate_cells
    from psm.covariates import build_resampled_covariates
    from psm.cell_features import extract_cells_with_covariates

    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    ee.data.setDeadline(EE_DEADLINE_MS)

    site_selector = SiteSelector()
    ee_crs_1km = ee.Projection(EE_CRS_METERS).atScale(PSM_CELL_SIZE)
    covariates = build_resampled_covariates(ee_crs_1km)

    fs = gcsfs.GCSFileSystem()
    inputs_modified = max(fs.modified(TREATMENT_CELLS), fs.modified(CONTROL_CELLS))

    failed = []
    for i, site_id in enumerate(site_ids, start=1):
        path = cell_covariates_path(site_id)
        print(f"\n{'=' * 70}\nSite {site_id} ({i}/{len(site_ids)})\n{'=' * 70}")
        if not overwrite and is_up_to_date(fs, path, inputs_modified):
            print(f"Up to date, skipping: {path}")
            continue
        try:
            pa_ctx = load_pa_candidate_cells(site_id, site_selector)
            _, cells_df = extract_cells_with_covariates(
                pa_ctx["grid_fc"], covariates, ee_crs_1km
            )
            # Sites with no cells are saved as empty tables, so they count as done
            cells_df.to_parquet(path, index=False)
            print(f"Saved {len(cells_df)} cells to {path}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append(site_id)

    n_done = len(site_ids) - len(failed)
    print(f"\nSite group {SITE_GROUP}: {n_done}/{len(site_ids)} sites done")
    if failed:
        print(f"Failed (re-run to retry): {failed}")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-extract every site"
    )
    extract_site_covariates(overwrite=parser.parse_args().overwrite)
