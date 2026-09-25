"""
Load candidate treatment and control cells for a single PA and build a labeled grid FeatureCollection.
"""

import pandas as pd
import geemap
import geopandas as gpd
import ee

from utils.variables import TREATMENT_CELLS, CONTROL_CELLS


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

    all_cells = gpd.GeoDataFrame(
        pd.concat([treatment_cells, control_cells], ignore_index=True)
    )
    if len(all_cells) == 0:
        grid_fc = ee.FeatureCollection([])
    else:
        all_cells["cell_ID"] = range(len(all_cells))
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
