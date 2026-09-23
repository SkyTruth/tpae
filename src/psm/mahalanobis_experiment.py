"""
Mahalanobis Distance Matching — Experiment
-------------------------------------------
Runs MDM iteratively on all 30 test sites and writes one site-level diagnostics CSV so experiment runs can be compared.
"""

# Change this for each experiment so results are not overwritten.
# Should be a unique identifier for the experiment so we know which changes caused which results.
run_id = "stratified_control_sampling"

from pathlib import Path
import sys
import os
import ee

cur = Path.cwd().resolve()
for parent in [cur] + list(cur.parents):
    if parent.name == "tpae":
        os.chdir(parent)
        break

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

from utils.variables import (
    PROJECT,
    EE_CRS_METERS,
    PSM_CELL_SIZE,
    TEST_SITE_IDS,
)

from absolute_effectiveness.site_selector import SiteSelector
from psm.prepare_pa_grid import load_pa_candidate_cells
from psm.covariates import build_resampled_covariates
from psm.cell_features import extract_cells_with_covariates
from psm.match_cells import match_treatment_control_mdm
from psm.diagnostics import site_diagnostics_row, save_experiment_diagnostics

ee.Authenticate()
ee.Initialize(project=PROJECT)

site_selector = SiteSelector()

EE_CRS_1km = ee.Projection(EE_CRS_METERS).atScale(PSM_CELL_SIZE)
output_path = f"results/mdm_experiments/{run_id}.csv"

# Load the covariate stack.
covariates = build_resampled_covariates(EE_CRS_1km)

# Iteratively apply MDM to each of the 30 test sites and record diagnostic results.

site_rows = []

for i, site_id in enumerate(TEST_SITE_IDS, start=1):
    print(f"\n{'=' * 70}")
    print(f"Site {site_id} ({i}/{len(TEST_SITE_IDS)})")
    print("=" * 70)
    try:
        pa_ctx = load_pa_candidate_cells(site_id, site_selector)
        grid_fc, cells_df = extract_cells_with_covariates(
            pa_ctx["grid_fc"], covariates, EE_CRS_1km
        )
        match_df, treat_df, control_df = match_treatment_control_mdm(cells_df)
        site_rows.append(site_diagnostics_row(match_df, treat_df, cells_df, site_id))
    except Exception as exc:
        print(f"FAILED: {exc}")
        site_rows.append({"site_id": site_id, "error": str(exc)})

    results_df = save_experiment_diagnostics(site_rows, output_path)

print("\nExperiment summary")
print(f"  Sites: {len(results_df)}")
print(f"  Mean match coverage: {results_df['match_coverage'].mean():.1%}")
print(f"  Mean |SMD| after: {results_df['avg_abs_smd_after'].mean():.3f}")
print(
    f"  Mean covariates balanced: {results_df['n_covariates_balanced'].mean():.1f} / 7"
)
print(f"\nSaved to {output_path}")
