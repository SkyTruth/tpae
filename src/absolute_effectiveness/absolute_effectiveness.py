"""
Absolute Effectiveness
-------------------------------------------
Computes habitat condition and loss metrics for every site in the active SITE_GROUP,
n_workers sites at a time, and writes one row per site to a CSV in GCS
(absolute_results/). Same analysis as notebooks/run_absolute_effectiveness.ipynb.

Transient Earth Engine errors are retried. Sites that still fail keep the columns
computed before the failure, with the reason in error. Use --retry-failed to re-run
only the sites that failed (or are missing) in the saved CSV.
Sites dropped for too few treatment cells (DROPPED_SITES, written by
get_treatment_cells.py and MDM_match_cells.py) are not analyzed; their error says why.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import sys
import threading

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd

from utils.variables import (
    PROJECT,
    EE_DEADLINE_MS,
    ANALYSIS_END_YR,
    GLC_LABELS,
    ABSOLUTE_RESULTS,
    DROPPED_SITES,
)

from sites.dropped_sites import load_dropped
from sites.site_selector import ee_geometry, load_sites
from utils.retries import (
    has_error,
    load_succeeded_rows,
    report_failures,
    with_retries,
)
from absolute_effectiveness.data_processor import DataProcessor
from absolute_effectiveness.habitat_condition import HabitatConditionAnalyzer
from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer

# After the imports, since habitat_loss.py initializes without a project
ee.Initialize(project=PROJECT)
ee.data.setDeadline(EE_DEADLINE_MS)

# Column prefix for each driver class in DRIVER_LABELS
DRIVER_COLUMNS = {
    1: "cropland",
    2: "built_up",
    3: "pasture",
    4: "deforestation",
    5: "tree_plantations",
}
N_HABITATS_LOST = 3
# Multiplier packing start land cover class and driver class into one group code
GROUP_BASE = 10
# Number of sites analyzed at once
N_WORKERS = 24
# Earth Engine requests in flight at once, under the project limit of 40
MAX_EE_REQUESTS = 36
EE_REQUESTS = threading.BoundedSemaphore(MAX_EE_REQUESTS)

COLUMNS = (
    [
        "site_id",
        "start_year",
        "end_year",
        "extent",
        "intactness",
        "condition",
        "loss",
        "loss_area_km2",
        "loss_pct_site",
        "loss_pct_habitat",
        "site_land_area_km2",
    ]
    + [
        f"{driver}_{metric}"
        for driver in DRIVER_COLUMNS.values()
        for metric in ("area_km2", "pct_site", "pct_habitat")
    ]
    + [
        f"habitat_lost_{i}{suffix}"
        for i in range(1, N_HABITATS_LOST + 1)
        for suffix in ("", "_area_km2", "_pct_start")
    ]
    + ["error"]
)

processor = DataProcessor.from_gee_defaults()
condition_analyzer = HabitatConditionAnalyzer()
loss_analyzer = HabitatLossAnalyzer()


def ee_request(fn, *args, **kwargs):
    """Call fn (one Earth Engine request) once a request slot is free."""
    with EE_REQUESTS:
        return fn(*args, **kwargs)


def analyze_site(site_id, START_YR, geometry):
    """Compute one site's results row, retrying transient errors; failed steps leave
    their columns blank.

    geometry: the site's cleaned (shapely) geometry from SITE_GEOMS.
    """
    row = {"site_id": site_id, "start_year": START_YR, "end_year": ANALYSIS_END_YR}
    try:
        with_retries(fill_site_row, row, START_YR, geometry, label=f"Site {site_id}")
    except Exception as exc:
        row["error"] = str(exc)
    return row


def fill_site_row(row, START_YR, geometry):
    """Add one site's results to row."""
    site_geom = ee_geometry(geometry)

    # Process input datasets
    GLC_processed = processor.process_glc(site_geom, START_YR)
    GPW_processed = processor.process_gpw(START_YR)
    FORTY_processed = processor.process_forty(site_geom, START_YR)
    HGFC_processed = processor.process_hgfc(START_YR)
    land_mask = processor.get_land_mask()

    # Build habitat rasters for the end and start years, and habitat loss
    lc_end = condition_analyzer.get_land_cover(
        GLC_processed, GPW_processed, FORTY_processed, ANALYSIS_END_YR
    )
    habitat_binary_end = condition_analyzer.get_habitat_binary_end(
        lc_end, HGFC_processed
    )
    lc_start = condition_analyzer.get_land_cover(
        GLC_processed, GPW_processed, FORTY_processed, START_YR
    )
    habitat_binary_start = condition_analyzer.get_habitat_binary_start(lc_start)
    habitat_loss_raster = loss_analyzer.get_habitat_loss_raster(
        habitat_binary_start, habitat_binary_end
    )

    # Build intactness raster
    exp_kernel = condition_analyzer.build_kernel()
    intactness_raster = condition_analyzer.get_intactness_raster(
        habitat_binary_end, site_geom, exp_kernel, land_mask
    )
    driver_class = loss_analyzer.get_driver_class_image(lc_end, habitat_loss_raster)
    # Group code: (start land cover class + 1) * GROUP_BASE + driver class
    group_image = (
        lc_start.add(1).unmask(0).multiply(GROUP_BASE).add(driver_class.unmask(0))
    )

    # Request intactness alongside the areas below
    with ThreadPoolExecutor(max_workers=1) as site_executor:
        intactness_future = site_executor.submit(
            ee_request,
            condition_analyzer.calc_intactness_score,
            intactness_raster,
            site_geom,
        )

        # Sum areas for extent, loss, drivers and habitat types lost in one request
        groups = ee_request(
            loss_analyzer.sum_areas_by_group_km2,
            site_geom,
            {
                "habitat_end": habitat_binary_end,
                "land": land_mask,
                "loss": habitat_loss_raster,
                "habitat_start": habitat_binary_start,
            },
            group_image,
        )
        areas = {
            name: sum(group[name] for group in groups.values())
            for name in ("habitat_end", "land", "loss", "habitat_start")
        }
        driver_areas, lost_areas, start_areas = {}, {}, {}
        for code, group in groups.items():
            start_class, driver = divmod(code, GROUP_BASE)
            if driver:
                driver_areas[driver] = driver_areas.get(driver, 0) + group["loss"]
            if start_class:
                lc_class = start_class - 1
                lost_areas[lc_class] = lost_areas.get(lc_class, 0) + group["loss"]
                start_areas[lc_class] = (
                    start_areas.get(lc_class, 0) + group["habitat_start"]
                )

        # Calculate Habitat Extent score
        row["extent"] = condition_analyzer.habitat_extent_from_areas(
            areas["habitat_end"], areas["land"]
        )

        # Calculate Habitat Loss score
        (
            row["loss_area_km2"],
            habitat_start_area_km2,
            row["site_land_area_km2"],
            row["loss_pct_site"],
            row["loss_pct_habitat"],
            row["loss"],
        ) = loss_analyzer.habitat_loss_from_areas(
            areas["loss"], areas["habitat_start"], areas["land"]
        )

        # Analyze drivers of habitat loss (absent drivers are 0)
        for class_id, driver in DRIVER_COLUMNS.items():
            area = driver_areas.get(class_id, 0) if habitat_start_area_km2 else 0
            row[f"{driver}_area_km2"] = area
            row[f"{driver}_pct_site"] = (
                area / row["site_land_area_km2"] * 100 if area else 0
            )
            row[f"{driver}_pct_habitat"] = (
                area / habitat_start_area_km2 * 100 if area else 0
            )

        # Analyze top types of habitat lost, largest first
        ranked = sorted(
            (c for c, area in lost_areas.items() if area > 0),
            key=lambda c: lost_areas[c],
            reverse=True,
        )[:N_HABITATS_LOST]
        for i, class_id in enumerate(ranked, start=1):
            row[f"habitat_lost_{i}"] = GLC_LABELS.get(class_id, f"Class {class_id}")
            row[f"habitat_lost_{i}_area_km2"] = lost_areas[class_id]
            row[f"habitat_lost_{i}_pct_start"] = (
                lost_areas[class_id] / start_areas[class_id] * 100
                if start_areas[class_id] > 0
                else 0
            )

        # Calculate Habitat Intactness and overall Habitat Condition scores
        row["intactness"] = intactness_future.result()
        row["condition"] = condition_analyzer.calc_habitat_condition_score(
            row["extent"], row["intactness"]
        )


def main(n_workers=N_WORKERS, retry_failed=False):
    """Analyze every site, or with retry_failed only those without saved results."""
    sites = load_sites()
    site_ids = [int(site_id) for site_id in sites["site_id"]]
    rows = load_succeeded_rows(ABSOLUTE_RESULTS, site_ids) if retry_failed else {}
    # Sites dropped for too few treatment cells are not analyzed
    dropped = load_dropped()
    for site_id, start_yr in zip(site_ids, sites["start_yr"]):
        if site_id in dropped:
            rows[site_id] = {
                "site_id": site_id,
                "start_year": int(start_yr),
                "end_year": ANALYSIS_END_YR,
                "error": f"dropped: {dropped[site_id]}",
            }
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyze_site, site_id, int(start_yr), geometry): site_id
            for site_id, start_yr, geometry in zip(
                site_ids, sites["start_yr"], sites.geometry
            )
            if site_id not in rows
        }
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            rows[site_id] = future.result()
            error = rows[site_id].get("error")
            status = f"FAILED: {error}" if error else "done"
            print(f"Site {site_id} ({i}/{len(futures)}): {status}")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            done = [rows[s] for s in site_ids if s in rows]
            pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(
                ABSOLUTE_RESULTS, index=False
            )

    done = [rows[s] for s in site_ids if s in rows]
    pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(ABSOLUTE_RESULTS, index=False)
    print(f"\nSaved {len(rows)} sites to {ABSOLUTE_RESULTS}")
    n_dropped = sum(site_id in dropped for site_id in site_ids)
    print(f"{n_dropped} site(s) dropped for too few treatment cells ({DROPPED_SITES})")
    errors = {
        s: rows[s]["error"]
        for s in site_ids
        if s in rows and has_error(rows[s]) and s not in dropped
    }
    report_failures(errors, len(rows) - n_dropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run only the sites that failed (or are missing) in the saved CSV",
    )
    main(retry_failed=parser.parse_args().retry_failed)
