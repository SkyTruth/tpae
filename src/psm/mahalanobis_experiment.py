"""
Mahalanobis Distance Matching — Experiment
-------------------------------------------
Runs MDM iteratively on every site in the active SITE_GROUP and, optionally, writes one
site-level diagnostics CSV (and report card) so experiment runs can be compared.

Reads each site's cell covariates from GCS, so run extract_cell_covariates.py first.
"""

# Change this for each experiment so results are not overwritten.
# Should be a unique identifier for the experiment so we know which changes caused which results.
run_id = "v3.1_holdout_set"
# Save the diagnostics CSV and report card to GCS; otherwise only print the summary
save_diagnostics = True

from pathlib import Path
import sys
import os
import pandas as pd

cur = Path.cwd().resolve()
for parent in [cur] + list(cur.parents):
    if parent.name == "tpae":
        os.chdir(parent)
        break

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

from utils.variables import (
    TEST_SITE_IDS,
    GCS_BUCKET,
    MDM_EXPERIMENTS_PREFIX,
    REPORT_CARDS_PREFIX,
)

from psm.extract_cell_covariates import cell_covariates_path
from psm.match_cells import match_treatment_control_mdm
from psm.diagnostics import (
    DIAGNOSTIC_COLUMNS,
    site_diagnostics_row,
    build_report_card,
    save_experiment_diagnostics,
    save_report_card,
)

output_path = f"gs://{GCS_BUCKET}/{MDM_EXPERIMENTS_PREFIX}{run_id}.csv"
report_card_path = f"gs://{GCS_BUCKET}/{REPORT_CARDS_PREFIX}{run_id}_report_card.csv"

# Iteratively apply MDM to each site and record diagnostic results.

site_rows = []

for i, site_id in enumerate(TEST_SITE_IDS, start=1):
    print(f"\n{'=' * 70}")
    print(f"Site {site_id} ({i}/{len(TEST_SITE_IDS)})")
    print("=" * 70)
    try:
        covariates_path = cell_covariates_path(site_id)
        try:
            cells_df = pd.read_parquet(covariates_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No cell covariates at {covariates_path}; "
                "run extract_cell_covariates.py first"
            ) from None
        match_df, treat_df, control_df = match_treatment_control_mdm(cells_df)
        site_rows.append(site_diagnostics_row(match_df, treat_df, cells_df, site_id))
    except Exception as exc:
        print(f"FAILED: {exc}")
        site_rows.append({"site_id": site_id, "error": str(exc)})

    # Rewrite the CSV after every site, so a stalled run keeps its results so far
    if save_diagnostics:
        save_experiment_diagnostics(site_rows, output_path)

results_df = pd.DataFrame(site_rows).reindex(columns=DIAGNOSTIC_COLUMNS)
if save_diagnostics:
    card = save_report_card(results_df, report_card_path)
else:
    card = build_report_card(results_df)
tiers = card["assessable"].value_counts()

print("\nExperiment summary")
print(f"  Sites: {len(results_df)}")
print("  " + ", ".join(f"{t}: {tiers.get(t, 0)}" for t in ("Yes", "Caution", "No")))
print(f"  Mean match coverage: {results_df['match_coverage'].mean():.1%}")
print(f"  Mean |SMD| after: {results_df['avg_abs_smd_after'].mean():.3f}")
print(
    f"  Mean covariates balanced: {results_df['n_covariates_balanced'].mean():.1f} / 7"
)
if save_diagnostics:
    print(f"\nSaved to {output_path} and {report_card_path}")
else:
    print("\nDiagnostics not saved (save_diagnostics = False)")
