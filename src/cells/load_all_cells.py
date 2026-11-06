"""
Load candidate treatment and control cells for a single site and build a FeatureCollection of them.
"""

import pandas as pd
import geemap
import geopandas as gpd
import ee

from utils.variables import TREATMENT_CELLS, CONTROL_CELLS


def build_site_cells(site_id, treatment_cells, control_cells):
    """A site's treatment then control cells, with sequential cell_IDs (EPSG:4326).

    treatment_cells / control_cells: cells from TREATMENT_CELLS / CONTROL_CELLS (all
    sites, or already filtered to this one). cell_IDs follow this order, so the same
    cell files always give the same IDs, including those in the extracted covariates.
    """
    site_id = str(site_id)
    site_cells = [
        cells[cells["site_id"] == site_id].to_crs(epsg=4326)
        for cells in (treatment_cells, control_cells)
    ]
    all_cells = gpd.GeoDataFrame(pd.concat(site_cells, ignore_index=True))
    all_cells["cell_ID"] = range(len(all_cells))
    return all_cells


def load_site_cells(site_id, site_selector):
    """Load treatment and control cells for a site and assign sequential cell_IDs."""
    site_id = str(site_id)

    sites = site_selector.get_sites()
    site_geom = site_selector.get_site_geom(sites, site_id)

    treatment_cells = gpd.read_parquet(TREATMENT_CELLS).to_crs(epsg=4326)
    treatment_cells = treatment_cells[treatment_cells["site_id"] == site_id]
    control_cells = gpd.read_parquet(CONTROL_CELLS).to_crs(epsg=4326)
    control_cells = control_cells[control_cells["site_id"] == site_id]

    print(f"Number of candidate treatment cells: {len(treatment_cells)}")
    print(f"Number of candidate control cells: {len(control_cells)}")

    all_cells = build_site_cells(site_id, treatment_cells, control_cells)
    if len(all_cells) == 0:
        cells_fc = ee.FeatureCollection([])
    else:
        all_cells["label"] = None
        cells_fc = geemap.geopandas_to_ee(all_cells)

    return {
        "site_id": site_id,
        "sites": sites,
        "site_geom": site_geom,
        "treatment_cells": treatment_cells,
        "control_cells": control_cells,
        "cells_fc": cells_fc,
    }
