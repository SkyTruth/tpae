"""
Absolute Effectiveness
-------------------------------------------
Computes habitat condition and loss metrics for every site in the active SITE_GROUP,
n_workers sites at a time, and writes one row per site to a CSV in GCS
(absolute_results/). Same analysis as notebooks/run_absolute_effectiveness.ipynb.

Sites that fail keep the columns computed before the failure, with the reason in error.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import ee
import pandas as pd

from utils.variables import (
    PROJECT,
    SITE_IDS,
    ANALYSIS_END_YR,
    GLC_LABELS,
    ABSOLUTE_RESULTS,
)

from sites.site_selector import SiteSelector
from absolute_effectiveness.data_processor import DataProcessor
from absolute_effectiveness.habitat_condition import HabitatConditionAnalyzer
from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer, parse_result_key

# After the imports, since habitat_loss.py initializes without a project
ee.Initialize(project=PROJECT)

# Column prefix for each driver class in DRIVER_LABELS
DRIVER_COLUMNS = {
    1: "cropland",
    2: "built_up",
    3: "pasture",
    4: "deforestation",
    5: "tree_plantations",
}
N_HABITATS_LOST = 3
# Number of sites analyzed at once
N_WORKERS = 24

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

site_selector = SiteSelector()
processor = DataProcessor.from_gee_defaults()
condition_analyzer = HabitatConditionAnalyzer()
loss_analyzer = HabitatLossAnalyzer()


def by_class(results_dict):
    """{class_id: {metric: value}} from a calc_*_area_and_pct result."""
    classes = {}
    for key, value in results_dict.items():
        parsed = parse_result_key(key)
        if parsed is not None:
            class_id, metric = parsed
            classes.setdefault(class_id, {})[metric] = value
    return classes


def analyze_site(site_id, sites, designation_yr):
    """Compute one site's results row; failed steps leave their columns blank.

    designation_yr: from get_designation_yrs (None if the site was not found).
    """
    row = {"site_id": site_id, "end_year": ANALYSIS_END_YR}
    try:
        # Get site-specific information
        if designation_yr is None:
            raise ValueError("Site not found among terrestrial PAs and OECMs")
        START_YR = site_selector.start_yr_from_designation(designation_yr)
        row["start_year"] = START_YR
        site_geom = site_selector.get_site_geom(sites, site_id)

        # Check if site designation year is valid for analysis
        site_selector.check_start_yr(START_YR)

        # Process input datasets
        GLC_processed = processor.process_glc(sites, START_YR)
        GPW_processed = processor.process_gpw(START_YR)
        FORTY_processed = processor.process_forty(sites, START_YR)
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
        habitat_class = loss_analyzer.get_habitat_class_image(
            lc_start, habitat_loss_raster
        )

        # Request intactness and habitat types lost alongside the areas and drivers below
        with ThreadPoolExecutor(max_workers=2) as site_executor:
            intactness_future = site_executor.submit(
                condition_analyzer.calc_intactness_score, intactness_raster, site_geom
            )
            habitats_future = site_executor.submit(
                loss_analyzer.calc_habitat_class_area_and_pct,
                habitat_class,
                habitat_binary_start,
                lc_start,
                site_geom,
                top_n=N_HABITATS_LOST,
            )

            # Sum the areas for the extent and loss scores in one request
            areas = loss_analyzer.sum_areas_km2(
                site_geom,
                {
                    "habitat_end": habitat_binary_end,
                    "land": land_mask,
                    "loss": habitat_loss_raster,
                    "habitat_start": habitat_binary_start,
                },
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

            # Analyze drivers of habitat loss (all classes; absent drivers are 0)
            driver_class = loss_analyzer.get_driver_class_image(
                lc_end, habitat_loss_raster
            )
            drivers = by_class(
                loss_analyzer.calc_class_area_and_pct(
                    driver_class,
                    site_geom,
                    row["site_land_area_km2"],
                    habitat_start_area_km2,
                    top_n=len(DRIVER_COLUMNS),
                )
            )
            for class_id, driver in DRIVER_COLUMNS.items():
                metrics = drivers.get(class_id, {})
                row[f"{driver}_area_km2"] = metrics.get("area", 0)
                row[f"{driver}_pct_site"] = metrics.get("pct_site", 0)
                row[f"{driver}_pct_habitat"] = metrics.get("pct_start", 0)

            # Calculate Habitat Intactness and overall Habitat Condition scores
            row["intactness"] = intactness_future.result()
            row["condition"] = condition_analyzer.calc_habitat_condition_score(
                row["extent"], row["intactness"]
            )

            # Analyze top types of habitat lost, largest first
            habitats = by_class(habitats_future.result())
        ranked = sorted(
            habitats, key=lambda c: habitats[c].get("area", 0), reverse=True
        )
        for i, class_id in enumerate(ranked, start=1):
            row[f"habitat_lost_{i}"] = GLC_LABELS.get(class_id, f"Class {class_id}")
            row[f"habitat_lost_{i}_area_km2"] = habitats[class_id].get("area")
            row[f"habitat_lost_{i}_pct_start"] = habitats[class_id].get("pct_class")
    except Exception as exc:
        row["error"] = str(exc)
    return row


def main(n_workers=N_WORKERS):
    sites = site_selector.get_sites()
    designation_yrs = site_selector.get_designation_yrs(sites)
    rows = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                analyze_site, site_id, sites, designation_yrs.get(site_id)
            ): site_id
            for site_id in SITE_IDS
        }
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            rows[site_id] = future.result()
            error = rows[site_id].get("error")
            status = f"FAILED: {error}" if error else "done"
            print(f"Site {site_id} ({i}/{len(SITE_IDS)}): {status}")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            done = [rows[s] for s in SITE_IDS if s in rows]
            pd.DataFrame(done).reindex(columns=COLUMNS).to_csv(
                ABSOLUTE_RESULTS, index=False
            )

    n_failed = sum("error" in row for row in rows.values())
    print(f"\nSaved {len(rows)} sites ({n_failed} failed) to {ABSOLUTE_RESULTS}")


if __name__ == "__main__":
    main()
