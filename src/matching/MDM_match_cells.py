"""
Mahalanobis Distance Matching
-------------------------------------------
Runs MDM on every site in the active SITE_GROUP, n_workers sites at a time (one process
each), and, optionally, writes one site-level diagnostics CSV (and report card), and
each site's match table and matched cells.

Reads each site's cell covariates from GCS, so run extract_cell_covariates.py first.
Sites dropped at earlier steps are skipped. Sites with fewer than MIN_TREATMENT_CELLS
treatment cells kept after covariate extraction, or matched, or that fail with an error,
are dropped (recorded in DROPPED_SITES); dropped sites' saved match outputs are deleted.
Use --sites to re-match only some sites, keeping the others' saved results.
"""

# Save the diagnostics CSV and report card to GCS; otherwise only print the summary
save_diagnostics = True
# Save each site's match table and matched cells to GCS (matches/<group>/), replacing any saved by an earlier run
save_match_outputs = True
# Number of sites matched at once, one process each (None = one per CPU core)
n_workers = None

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import argparse
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
from sites.dropped_sites import (
    failed_reason,
    load_dropped,
    load_dropped_before,
    record_dropped,
)
from cells.load_all_cells import build_all_site_cells
from matching.match_cells import (
    match_output_paths,
    match_treatment_control_mdm,
    save_matching_outputs,
    scored_pairs_path,
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
            dropped = failed_reason(exc)
    return row, dropped, log.getvalue()


def main(listed=None):
    """Match every site, or only the listed sites (keeping the others' saved results)."""
    site_ids = load_site_ids()
    # Sites dropped before matching, and those dropped here
    dropped_before = load_dropped_before("matching")
    dropped_before = {s: dropped_before[s] for s in site_ids if s in dropped_before}
    dropped = {}
    rows = {site_id: {"site_id": site_id} for site_id in dropped_before}
    to_match = [site_id for site_id in site_ids if site_id not in dropped_before]
    # Other sites' saved diagnostics and matching drops
    others_dropped = {}
    if listed is not None:
        saved = pd.read_csv(MATCH_DIAGNOSTICS).to_dict("records")
        rows.update(
            {
                int(row["site_id"]): row
                for row in saved
                if int(row["site_id"]) not in listed
                and int(row["site_id"]) not in dropped_before
            }
        )
        others_dropped = {
            s: reason
            for s, reason in load_dropped(["matching"]).items()
            if s not in listed
        }
        to_match = [site_id for site_id in to_match if site_id in listed]
    # Cell geometries for the saved matched cells (cell_IDs are rebuilt from these files)
    cells_by_site = {}
    if save_match_outputs:
        cells = build_all_site_cells(
            site_ids if listed is None else to_match,
            gpd.read_parquet(TREATMENT_CELLS),
            gpd.read_parquet(CONTROL_CELLS),
        )
        cells_by_site = dict(tuple(cells.groupby("site_id")))

    # Apply MDM to each site and record diagnostic results
    print(
        f"Skipping {len(dropped_before)} site(s) dropped at earlier steps; "
        f"matching {len(to_match)}"
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                match_site, site_id, cells_by_site.get(str(site_id))
            ): site_id
            for site_id in to_match
        }
        for i, future in enumerate(as_completed(futures), start=1):
            site_id = futures[future]
            try:
                rows[site_id], reason, log = future.result()
            except Exception as exc:
                # The site's process failed, e.g. ran out of memory
                rows[site_id] = {"site_id": site_id, "error": str(exc)}
                reason, log = failed_reason(exc), f"FAILED: {exc}\n"
            if reason:
                dropped[site_id] = reason
            print(f"\n{'=' * 70}\nSite {site_id} ({i}/{len(to_match)})\n{'=' * 70}")
            print(log, end="")

            # Rewrite the CSV after every site, so a stalled run keeps its results so far
            if save_diagnostics:
                done = [rows[s] for s in site_ids if s in rows]
                save_experiment_diagnostics(done, MATCH_DIAGNOSTICS)

    record_dropped("matching", dropped, sites=listed)
    all_dropped = {**dropped_before, **others_dropped, **dropped}
    # Delete dropped sites' match outputs from earlier runs, so they are not used
    fs = gcsfs.GCSFileSystem()
    for site_id in all_dropped:
        if listed is not None and site_id not in listed:
            continue
        for path in (*match_output_paths(site_id), scored_pairs_path(site_id)):
            if fs.exists(path):
                fs.rm(path)

    site_rows = [rows[site_id] for site_id in site_ids]
    results_df = pd.DataFrame(site_rows).reindex(columns=DIAGNOSTIC_COLUMNS)
    if save_diagnostics:
        card = save_report_card(results_df, MATCH_REPORT_CARD, all_dropped)
    else:
        card = build_report_card(results_df, all_dropped)
    # Summary statistics over the sites kept for analysis
    results_df = results_df[~results_df["site_id"].isin(all_dropped)]
    tiers = card.loc[~card["site_id"].isin(all_dropped), "assessable"].value_counts()

    print(f"\nExperiment summary ({len(results_df)} kept sites)")
    print(
        "  Report card: "
        + ", ".join(f"{t}: {tiers.get(t, 0)}" for t in ("Yes", "Caution", "No"))
    )
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites",
        nargs="+",
        type=int,
        metavar="SITE_ID",
        help="re-match only these sites, keeping the other sites' saved results",
    )
    main(listed=parser.parse_args().sites)
