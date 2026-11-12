"""
Mahalanobis Distance Matching
-------------------------------------------
Runs MDM on every site in the active SITE_GROUP, n_workers sites at a time (one process
each), and, optionally, writes one site-level diagnostics CSV (and report card), and
each site's match table and matched cells.

Reads each site's cell covariates from GCS, so run extract_cell_covariates.py first.
Sites dropped by get_treatment_cells.py are skipped, and sites with fewer than
MIN_TREATMENT_CELLS treatment cells kept after covariate extraction, or matched, are
dropped (recorded in DROPPED_SITES); dropped sites' saved match outputs are deleted.
"""

# Save the diagnostics CSV and report card to GCS; otherwise only print the summary
save_diagnostics = True
# Save each site's match table and matched cells to GCS (matches/<group>/), replacing any saved by an earlier run
save_match_outputs = True
# Number of sites matched at once, one process each (None = one per CPU core)
n_workers = None

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import os
import gcsfs
import geopandas as gpd
import pandas as pd

cur = Path.cwd().resolve()
for parent in [cur] + list(cur.parents):
    if parent.name == "tpae":
        os.chdir(parent)
        break

sys.path.insert(0, str((Path.cwd() / "src").resolve()))

from utils.variables import (
    GCS_BUCKET,
    MATCH_DIAGNOSTICS,
    MATCH_REPORT_CARD,
    MATCHES_PREFIX,
    TREATMENT_CELLS,
    CONTROL_CELLS,
    MIN_TREATMENT_CELLS,
)

from covariates.extract_cell_covariates import cell_covariates_path
from sites.site_selector import load_site_ids
from sites.dropped_sites import load_dropped, record_dropped
from cells.load_all_cells import build_all_site_cells
from matching.match_cells import (
    match_output_paths,
    match_treatment_control_mdm,
    save_matching_outputs,
)
from matching.diagnostics import (
    DIAGNOSTIC_COLUMNS,
    site_diagnostics_row,
    build_report_card,
    save_experiment_diagnostics,
    save_report_card,
)

matches_folder = f"gs://{GCS_BUCKET}/{MATCHES_PREFIX}"


def match_site(site_id, site_cells):
    """Match one site; return its diagnostics row, why it was dropped (None if kept),
    and what it printed.

    site_cells: the site's cells with cell_IDs, for the saved matched cells (None if
    match outputs are not saved).
    """
    log = StringIO()
    dropped = None
    with redirect_stdout(log):
        try:
            covariates_path = cell_covariates_path(site_id)
            try:
                cells_df = pd.read_parquet(covariates_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"No cell covariates at {covariates_path}; "
                    "run extract_cell_covariates.py first"
                ) from None
            n_kept = int((cells_df["protected"] == 1).sum()) if len(cells_df) else 0
            if n_kept < MIN_TREATMENT_CELLS:
                dropped = (
                    f"fewer than {MIN_TREATMENT_CELLS} treatment cells kept after "
                    f"covariate extraction ({n_kept})"
                )
                row = {"site_id": site_id}
            else:
                match_df, treat_df, control_df = match_treatment_control_mdm(cells_df)
                row = site_diagnostics_row(match_df, treat_df, cells_df, site_id)
                n_matched = int(row["n_matched_treat"])
                if n_matched < MIN_TREATMENT_CELLS:
                    dropped = (
                        f"fewer than {MIN_TREATMENT_CELLS} matched treatment cells "
                        f"({n_matched})"
                    )
                elif save_match_outputs:
                    save_matching_outputs(site_cells, cells_df, match_df, site_id)
            if dropped:
                print(f"Dropped: {dropped}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            row = {"site_id": site_id, "error": str(exc)}
    return row, dropped, log.getvalue()


def main():
    site_ids = load_site_ids()
    # Sites dropped before matching, and those dropped here
    dropped_before = load_dropped(steps=["treatment_cells"])
    dropped = {}
    # Cell geometries for the saved matched cells (cell_IDs are rebuilt from these files)
    cells_by_site = {}
    if save_match_outputs:
        cells = build_all_site_cells(
            site_ids, gpd.read_parquet(TREATMENT_CELLS), gpd.read_parquet(CONTROL_CELLS)
        )
        cells_by_site = dict(tuple(cells.groupby("site_id")))

    # Apply MDM to each site and record diagnostic results
    rows = {site_id: {"site_id": site_id} for site_id in dropped_before}
    for site_id, reason in dropped_before.items():
        print(f"Skipping site {site_id}: dropped at treatment cells ({reason})")
    to_match = [site_id for site_id in site_ids if site_id not in dropped_before]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                match_site, site_id, cells_by_site.get(str(site_id))
            ): site_id
            for site_id in to_match
        }
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            rows[site_id], reason, log = future.result()
            if reason:
                dropped[site_id] = reason
            print(f"\n{'=' * 70}\nSite {site_id} ({i}/{len(to_match)})\n{'=' * 70}")
            print(log, end="")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            if save_diagnostics:
                done = [rows[s] for s in site_ids if s in rows]
                save_experiment_diagnostics(done, MATCH_DIAGNOSTICS)

    record_dropped("matching", dropped)
    all_dropped = {**dropped_before, **dropped}
    # Delete dropped sites' match outputs from earlier runs, so they are not used
    fs = gcsfs.GCSFileSystem()
    for site_id in all_dropped:
        for path in match_output_paths(site_id):
            if fs.exists(path):
                fs.rm(path)

    site_rows = [rows[site_id] for site_id in site_ids]
    results_df = pd.DataFrame(site_rows).reindex(columns=DIAGNOSTIC_COLUMNS)
    if save_diagnostics:
        card = save_report_card(results_df, MATCH_REPORT_CARD, all_dropped)
    else:
        card = build_report_card(results_df, all_dropped)
    tiers = card["assessable"].value_counts()
    # Summary statistics over the sites kept for analysis
    results_df = results_df[~results_df["site_id"].isin(all_dropped)]

    print("\nExperiment summary")
    print(f"  Sites: {len(results_df)} kept, {len(all_dropped)} dropped")
    for site_id, reason in all_dropped.items():
        print(f"    dropped {site_id}: {reason}")
    print("  " + ", ".join(f"{t}: {tiers.get(t, 0)}" for t in ("Yes", "Caution", "No")))
    print(f"  Mean match coverage: {results_df['match_coverage'].mean():.1%}")
    print(f"  Mean |SMD| after: {results_df['avg_abs_smd_after'].mean():.3f}")
    print(
        f"  Mean covariates balanced: {results_df['n_covariates_balanced'].mean():.1f} / 7"
    )
    if save_diagnostics:
        print(f"\nSaved to {MATCH_DIAGNOSTICS} and {MATCH_REPORT_CARD}")
    else:
        print("\nDiagnostics not saved (save_diagnostics = False)")
    if save_match_outputs:
        print(f"Match outputs saved to {matches_folder}")


if __name__ == "__main__":
    main()
