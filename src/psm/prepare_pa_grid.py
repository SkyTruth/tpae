"""
Load candidate treatment and control cells for a single PA and build a labeled grid FeatureCollection.
"""

import ee
import geemap
import geopandas as gpd

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

    treatment_fc = geemap.geopandas_to_ee(treatment_cells)
    control_fc = geemap.geopandas_to_ee(control_cells)
    all_cells = ee.FeatureCollection([treatment_fc, control_fc]).flatten()

    cell_IDs = ee.List.sequence(0, all_cells.size().getInfo() - 1)
    featureList = all_cells.toList(all_cells.size())
    grid_fc = ee.FeatureCollection(
        cell_IDs.map(
            lambda cell_ID: ee.Feature(featureList.get(cell_ID)).set(
                {"cell_ID": cell_ID, "label": None}
            )
        )
    )

    return {
        "PA_ID": PA_ID,
        "test_sites": test_sites,
        "site_geom": site_geom,
        "treatment_cells": treatment_cells,
        "control_cells": control_cells,
        "grid_fc": grid_fc,
    }
