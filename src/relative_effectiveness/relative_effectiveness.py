"""
Relative Effectiveness
-------------------------------------------
Scores each site's matched treatment and control cells and compares them, for every
site in the active SITE_GROUP, n_workers sites at a time, and writes one row of
site-level relative scores per site to a CSV in GCS (relative_results/), and each
site's scored matched pairs next to its match outputs (scored_pairs_path). Same
analysis as notebooks/run_relative_effectiveness.ipynb.

Reads each site's match outputs from GCS, so run MDM_match_cells.py with
save_match_outputs = True first. Transient Earth Engine errors are retried. Sites that
still fail keep the columns computed before the failure, with the reason in error, and
are dropped (recorded in DROPPED_SITES). Use --retry-failed to re-run only the sites that
failed (or are missing) in the saved CSV, or --sites to re-run only some sites. Sites
dropped at earlier steps are not analyzed; their error says why.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import geemap
import geopandas as gpd
import pandas as pd

from utils.variables import (
    PROJECT,
    ANALYSIS_END_YR,
    RELATIVE_RESULTS,
    KERNEL_RADIUS_M,
)

from sites.dropped_sites import failed_reason, load_dropped_before, record_dropped
from sites.site_selector import bounds_region, load_start_yrs
from utils.retries import (
    has_error,
    load_succeeded_rows,
    report_failures,
    set_request_limits,
    with_retries,
)
from absolute_effectiveness.data_processor import DataProcessor
from matching.match_cells import match_output_paths, scored_pairs_path
from relative_effectiveness.metrics_per_cell import (
    RelativeHabitatConditionAnalyzer,
    RelativeHabitatLossAnalyzer,
)
from relative_effectiveness.relative_scores import (
    SCORE_COLS,
    START_HABITAT_COL,
    build_match_df_with_score_diffs,
    regression_adjust_score_diffs,
    aggregate_site_relative_scores,
)

# Before the Earth Engine objects below are built
ee.Initialize(project=PROJECT)
set_request_limits()

SCORE_NAMES = ["extent", "intactness", "condition", "loss"]
COLUMNS = (
    ["site_id", "start_year", "end_year"]
    + [f"{name}_{kind}" for name in SCORE_NAMES for kind in ("unadj", "adj")]
    + ["mean_pair_habitat_weight", "regression_adjusted", "error"]
)
# Number of sites analyzed at once
N_WORKERS = 24

processor = DataProcessor.from_gee_defaults()
condition_analyzer = RelativeHabitatConditionAnalyzer()
loss_analyzer = RelativeHabitatLossAnalyzer()


def analyze_site(site_id, START_YR):
    """Compute one site's results row, retrying transient errors; failed steps leave
    their columns blank."""
    row = {"site_id": site_id, "start_year": START_YR, "end_year": ANALYSIS_END_YR}
    try:
        with_retries(fill_site_row, row, site_id, START_YR, label=f"Site {site_id}")
    except Exception as exc:
        row["error"] = str(exc)
    return row


def fill_site_row(row, site_id, START_YR):
    """Add one site's results to row."""
    # Load matched_cells and the match table (saved match outputs in GCS), both at once
    match_table_path, matched_cells_path = match_output_paths(site_id)
    try:
        with ThreadPoolExecutor(max_workers=2) as read_executor:
            cells_future = read_executor.submit(gpd.read_parquet, matched_cells_path)
            table_future = read_executor.submit(pd.read_parquet, match_table_path)
            matched_cells_gdf = cells_future.result().to_crs(epsg=4326)
            match_table = table_future.result().astype(
                {"treat_cell_id": int, "control_cell_id": int}
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No match outputs at {matched_cells_path}; run MDM_match_cells.py "
            "with save_match_outputs = True"
        ) from None
    matched_cells = geemap.geopandas_to_ee(matched_cells_gdf)
    # Input images are read within the cells and the intactness kernel's reach
    inputs_region = bounds_region(matched_cells_gdf.geometry.values, KERNEL_RADIUS_M)

    # Process input datasets
    GLC_processed = processor.process_glc(inputs_region, START_YR)
    GPW_processed = processor.process_gpw(START_YR)
    FORTY_processed = processor.process_forty(inputs_region, START_YR)
    HGFC_processed = processor.process_hgfc(START_YR)
    land_mask = processor.get_land_mask()

    lc_end = condition_analyzer.get_land_cover(
        GLC_processed, GPW_processed, FORTY_processed, ANALYSIS_END_YR
    )
    lc_start = condition_analyzer.get_land_cover(
        GLC_processed, GPW_processed, FORTY_processed, START_YR
    )
    habitat_binary_end = condition_analyzer.get_habitat_binary_end(
        lc_end, HGFC_processed
    )
    habitat_binary_start = condition_analyzer.get_habitat_binary_start(lc_start)

    # Build intactness raster
    exp_kernel = condition_analyzer.build_kernel()
    intactness_raster = condition_analyzer.get_intactness_raster(
        habitat_binary_end, exp_kernel, land_mask
    )

    # Build habitat-loss raster
    habitat_loss_raster = loss_analyzer.get_habitat_loss_raster(
        habitat_binary_start, habitat_binary_end
    )

    # Score Habitat Extent and Habitat Loss per cell, with the start-habitat
    # proportion that weights matched pairs' loss differences, in one pass
    scored = loss_analyzer.calc_area_scores_per_cell(
        habitat_binary_end, habitat_loss_raster, habitat_binary_start, matched_cells
    )
    scored = condition_analyzer.calc_intactness_score_per_cell(
        intactness_raster, scored
    )
    scored = condition_analyzer.calc_condition_score_per_cell(scored)

    # Download the scores once (same as geemap.ee_to_gdf, without its extra
    # request for the CRS)
    scored_gdf = ee.data.computeFeatures(
        {"expression": scored, "fileFormat": "GEOPANDAS_GEODATAFRAME"}
    ).set_crs(epsg=4326)

    # Calculate relative effectiveness, unadjusted and regression adjusted
    match_df = build_match_df_with_score_diffs(
        scored_gdf[["geometry", "cell_ID", *SCORE_COLS, START_HABITAT_COL]],
        match_table,
    )
    # Covariates for each matched cell come from the saved matched cells
    match_df = regression_adjust_score_diffs(match_df, matched_cells_gdf)
    row["regression_adjusted"] = match_df.attrs["regression_adjusted"]
    # Save each pair's scores and differences
    match_df.to_parquet(scored_pairs_path(site_id), index=False)

    _, site_scores = aggregate_site_relative_scores(match_df)
    for name in SCORE_NAMES:
        row[f"{name}_unadj"] = site_scores[name]
        row[f"{name}_adj"] = site_scores[f"{name}_adj"]
    row["mean_pair_habitat_weight"] = site_scores["mean_pair_habitat_weight"]


def main(n_workers=N_WORKERS, retry_failed=False, sites=None):
    """Analyze every site, with retry_failed only those without saved results, or with
    sites only those sites (keeping the others' saved rows)."""
    start_yrs = load_start_yrs()
    site_ids = list(start_yrs)
    rows = load_succeeded_rows(RELATIVE_RESULTS, site_ids) if retry_failed else {}
    if sites is not None:
        saved = pd.read_csv(RELATIVE_RESULTS).to_dict("records")
        rows = {int(row["site_id"]): row for row in saved}
        rows = {s: row for s, row in rows.items() if s in start_yrs and s not in sites}
    # Sites dropped at earlier steps are not analyzed
    dropped = load_dropped_before("relative_effectiveness")
    for site_id, start_yr in start_yrs.items():
        if site_id in dropped:
            rows[site_id] = {
                "site_id": site_id,
                "start_year": int(start_yr),
                "end_year": ANALYSIS_END_YR,
                "error": f"dropped: {dropped[site_id]}",
            }
    n_dropped = sum(site_id in dropped for site_id in site_ids)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyze_site, site_id, start_yr): site_id
            for site_id, start_yr in start_yrs.items()
            if site_id not in rows
        }
        print(
            f"Skipping {n_dropped} site(s) dropped at earlier steps; "
            f"analyzing {len(futures)}"
        )
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            rows[site_id] = future.result()
            error = rows[site_id].get("error")
            status = f"FAILED: {error}" if error else "done"
            print(f"Site {site_id} ({i}/{len(futures)}): {status}")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            done = [rows[s] for s in site_ids if s in rows]
            pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(
                RELATIVE_RESULTS, index=False
            )

    done = [rows[s] for s in site_ids if s in rows]
    pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(RELATIVE_RESULTS, index=False)
    print(
        f"\nSaved {len(rows)} rows ({len(rows) - n_dropped} analyzed sites, "
        f"{n_dropped} dropped at earlier steps) to {RELATIVE_RESULTS}"
    )
    errors = {
        s: rows[s]["error"]
        for s in site_ids
        if s in rows and has_error(rows[s]) and s not in dropped
    }
    report_failures(errors, len(rows) - n_dropped)
    record_dropped(
        "relative_effectiveness", {s: failed_reason(e) for s, e in errors.items()}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run only the sites that failed (or are missing) in the saved CSV",
    )
    which.add_argument(
        "--sites",
        nargs="+",
        type=int,
        metavar="SITE_ID",
        help="re-run only these sites, keeping the other sites' saved rows",
    )
    args = parser.parse_args()
    main(retry_failed=args.retry_failed, sites=args.sites)
