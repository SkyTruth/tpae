"""
Relative Effectiveness
-------------------------------------------
Scores each site's matched treatment and control cells and compares them, for every
site in the active SITE_GROUP, n_workers sites at a time, and writes one row of
site-level relative scores per site to a CSV in GCS (relative_results/). Same analysis
as notebooks/run_relative_effectiveness.ipynb.

Reads each site's match outputs from GCS, so run MDM_match_cells.py with
save_match_outputs = True first. Sites that fail keep the columns computed before the
failure, with the reason in error.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
)

from sites.site_selector import load_start_yrs
from absolute_effectiveness.data_processor import DataProcessor
from matching.match_cells import match_output_paths
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

# After the imports, since habitat_loss.py initializes without a project
ee.Initialize(project=PROJECT)

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
    """Compute one site's results row; failed steps leave their columns blank."""
    row = {"site_id": site_id, "start_year": START_YR, "end_year": ANALYSIS_END_YR}
    try:
        # Load matched_cells and the match table (saved match outputs in GCS), both at once
        match_table_path, matched_cells_path = match_output_paths(site_id)
        try:
            with ThreadPoolExecutor(max_workers=2) as read_executor:
                cells_future = read_executor.submit(
                    gpd.read_parquet, matched_cells_path
                )
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

        # Process input datasets
        GLC_processed = processor.process_glc(matched_cells, START_YR)
        GPW_processed = processor.process_gpw(START_YR)
        FORTY_processed = processor.process_forty(matched_cells, START_YR)
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
            habitat_binary_end,
            matched_cells.geometry(),
            exp_kernel,
            land_mask,
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

        _, site_scores = aggregate_site_relative_scores(match_df)
        for name in SCORE_NAMES:
            row[f"{name}_unadj"] = site_scores[name]
            row[f"{name}_adj"] = site_scores[f"{name}_adj"]
        row["mean_pair_habitat_weight"] = site_scores["mean_pair_habitat_weight"]
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main(n_workers=N_WORKERS):
    start_yrs = load_start_yrs()
    site_ids = list(start_yrs)
    rows = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyze_site, site_id, start_yr): site_id
            for site_id, start_yr in start_yrs.items()
        }
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            rows[site_id] = future.result()
            error = rows[site_id].get("error")
            status = f"FAILED: {error}" if error else "done"
            print(f"Site {site_id} ({i}/{len(site_ids)}): {status}")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            done = [rows[s] for s in site_ids if s in rows]
            pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(
                RELATIVE_RESULTS, index=False
            )

    n_failed = sum("error" in row for row in rows.values())
    print(f"\nSaved {len(rows)} sites ({n_failed} failed) to {RELATIVE_RESULTS}")


if __name__ == "__main__":
    main()
