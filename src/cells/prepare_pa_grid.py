"""
Load candidate treatment and control cells for a single PA and build a labeled grid FeatureCollection.
"""

import pandas as pd
import geemap
import geopandas as gpd
import ee

from utils.variables import TREATMENT_CELLS, CONTROL_CELLS


def build_site_cells(site_id, treatment_cells, control_cells):
    """A PA's treatment then control cells, with sequential cell_IDs (EPSG:4326).

    treatment_cells / control_cells: cells from TREATMENT_CELLS / CONTROL_CELLS (all
    sites, or already filtered to this one). cell_IDs follow this order, so the same
    cell files always give the same IDs, including those in the extracted covariates.
    """
    PA_ID = str(site_id)
    site_cells = [
        cells[cells["WDPAID"] == PA_ID].to_crs(epsg=4326)
        for cells in (treatment_cells, control_cells)
    ]
    all_cells = gpd.GeoDataFrame(pd.concat(site_cells, ignore_index=True))
    all_cells["cell_ID"] = range(len(all_cells))
    return all_cells


def load_pa_candidate_cells(site_id, site_selector):
    """Load treatment and control cells for a PA and assign sequential cell_IDs."""
    PA_ID = str(site_id)

    test_sites = site_selector.get_test_sites()
    site_geom = site_selector.get_site_geom(test_sites, site_id)

    treatment_cells = gpd.read_parquet(TREATMENT_CELLS).to_crs(epsg=4326)
    treatment_cells = treatment_cells[treatment_cells["WDPAID"] == PA_ID]
    control_cells = gpd.read_parquet(CONTROL_CELLS).to_crs(epsg=4326)
    control_cells = control_cells[control_cells["WDPAID"] == PA_ID]

    print(f"Number of candidate treatment cells: {len(treatment_cells)}")
    print(f"Number of candidate control cells: {len(control_cells)}")

    all_cells = build_site_cells(site_id, treatment_cells, control_cells)
    if len(all_cells) == 0:
        grid_fc = ee.FeatureCollection([])
    else:
        all_cells["label"] = None
        grid_fc = geemap.geopandas_to_ee(all_cells)

    return {
        "PA_ID": PA_ID,
        "test_sites": test_sites,
        "site_geom": site_geom,
        "treatment_cells": treatment_cells,
        "control_cells": control_cells,
        "grid_fc": grid_fc,
    }
