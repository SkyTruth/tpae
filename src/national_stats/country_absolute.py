"""
Absolute effectiveness of a country's protected land: the habitat condition and loss
results absolute_effectiveness.py computes for a site, for every PA pixel in the
country, reduced over the country in grid pieces (site_chunks).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from absolute_effectiveness.absolute_effectiveness import (
    COLUMNS as SITE_COLUMNS,
    DRIVER_COLUMNS,
    GROUP_BASE,
    MAX_EE_REQUESTS,
    N_HABITATS_LOST,
)
from absolute_effectiveness.data_processor import DataProcessor
from absolute_effectiveness.habitat_condition import HabitatConditionAnalyzer
from absolute_effectiveness.habitat_loss import HabitatLossAnalyzer
from sites.site_selector import bounds_rectangle, site_chunks
from utils.retries import with_retries
from utils.variables import ANALYSIS_END_YR, GLC_LABELS, KERNEL_RADIUS_M

# Results columns: absolute_effectiveness.py's, with country in place of site_id
COLUMNS = ["country", *SITE_COLUMNS[1:]]
# Masks whose areas are summed per group
AREA_NAMES = ("habitat_end", "land", "loss", "habitat_start")

processor = DataProcessor.from_gee_defaults()
condition_analyzer = HabitatConditionAnalyzer()
loss_analyzer = HabitatLossAnalyzer()


def country_rasters(inputs_region, start_yr):
    """The rasters absolute_effectiveness.py builds for a site, from input images that
    overlap inputs_region, by name (server side)."""
    glc = processor.process_glc(inputs_region, start_yr)
    gpw = processor.process_gpw(start_yr)
    forty = processor.process_forty(inputs_region, start_yr)
    hgfc = processor.process_hgfc(start_yr)
    land = processor.get_land_mask()

    lc_end = condition_analyzer.get_land_cover(glc, gpw, forty, ANALYSIS_END_YR)
    habitat_end = condition_analyzer.get_habitat_binary_end(lc_end, hgfc)
    lc_start = condition_analyzer.get_land_cover(glc, gpw, forty, start_yr)
    habitat_start = condition_analyzer.get_habitat_binary_start(lc_start)
    loss = loss_analyzer.get_habitat_loss_raster(habitat_start, habitat_end)
    intactness = condition_analyzer.get_intactness_raster(
        habitat_end, condition_analyzer.build_kernel(), land
    )
    driver_class = loss_analyzer.get_driver_class_image(lc_end, loss)
    # Group code: (start land cover class + 1) * GROUP_BASE + driver class
    group = lc_start.add(1).unmask(0).multiply(GROUP_BASE).add(driver_class.unmask(0))
    return {
        "glc": glc,
        "gpw": gpw,
        "forty": forty,
        "hgfc": hgfc,
        "land": land,
        "lc_end": lc_end,
        "lc_start": lc_start,
        "habitat_end": habitat_end,
        "habitat_start": habitat_start,
        "loss": loss,
        "intactness": intactness,
        "driver_class": driver_class,
        "group": group,
    }


def merge_groups(piece_groups):
    """Each group's areas summed over pieces ({group: {name: area}})."""
    total = {}
    for groups in piece_groups:
        for code, areas in groups.items():
            group_total = total.setdefault(code, dict.fromkeys(AREA_NAMES, 0))
            for name in AREA_NAMES:
                group_total[name] += areas[name]
    return total


def fill_absolute_row(row, groups, intactness_sums):
    """Add the results of absolute_effectiveness.py to row, from area sums by group
    (sum_areas_by_group_km2) and intactness sums (intactness_sums) of every piece."""
    areas = {name: sum(group[name] for group in groups.values()) for name in AREA_NAMES}
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
    row["intactness"] = condition_analyzer.intactness_from_sums(intactness_sums)
    row["condition"] = condition_analyzer.calc_habitat_condition_score(
        row["extent"], row["intactness"]
    )


def analyze_country(country, start_yr, country_shape, pas, n_workers=MAX_EE_REQUESTS):
    """One results row (COLUMNS) for all PA pixels in the country.

    country_shape: the country's shapely geometry (EPSG:4326), cut into site_chunks
    pieces; pas: the PAs whose pixels are analyzed (not clipped to the country). Two
    requests per piece (areas and intactness), n_workers at a time.
    """
    row = {"country": country, "start_year": start_yr, "end_year": ANALYSIS_END_YR}
    pieces = site_chunks(country_shape)
    # Input images are read within the country and the intactness kernel's reach
    inputs_region = bounds_rectangle(country_shape.bounds, KERNEL_RADIUS_M)
    rasters = country_rasters(inputs_region, start_yr)
    masks = {name: rasters[name] for name in AREA_NAMES}

    def piece_areas(piece):
        # Areas within the PA polygons in the piece
        return loss_analyzer.sum_areas_by_group_km2(
            (pas.filterBounds(piece), piece), masks, rasters["group"]
        )

    def piece_intactness(piece):
        piece_pas = pas.filterBounds(piece)
        return condition_analyzer.intactness_sums(
            rasters["intactness"].clipToCollection(piece_pas), piece
        )

    jobs = [
        (fn, i) for i in range(len(pieces)) for fn in (piece_areas, piece_intactness)
    ]
    print(
        f"{country}: {len(pieces)} piece(s), {len(jobs)} requests, "
        f"{n_workers} at a time"
    )
    piece_groups, intactness_sums = [], []
    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        futures = {
            executor.submit(
                with_retries, fn, pieces[i], label=f"Piece {i} ({fn.__name__})"
            ): fn
            for fn, i in jobs
        }
        for n_done, future in enumerate(as_completed(futures), start=1):
            if futures[future] is piece_areas:
                piece_groups.append(future.result())
            else:
                intactness_sums.append(future.result())
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"  {n_done}/{len(jobs)} requests done")
    finally:
        # Stop unstarted requests if one failed or was interrupted
        executor.shutdown(cancel_futures=True)

    fill_absolute_row(row, merge_groups(piece_groups), intactness_sums)
    return row
