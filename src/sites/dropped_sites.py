"""
Record and read sites dropped from the analysis (DROPPED_SITES, one row per site with
the step that dropped it and why). A step drops sites that fail one of its rules (e.g.
too few treatment cells) and sites that failed with an error (reason "failed: <error>").

Each step skips the sites dropped at earlier steps. Recording a step clears the records
of later steps, which are out of date until those steps re-run, and prints how many of
the group's sites were dropped before and at the step, and how many remain.
"""

import re

import pandas as pd

from utils.variables import DROPPED_SITES, SITE_GEOMS

COLUMNS = ["site_id", "step", "reason"]
# Steps that drop sites, in pipeline order
STEPS = [
    "clean_sites",
    "treatment_cells",
    "control_cells",
    "covariates",
    "matching",
    "relative_effectiveness",
    "absolute_effectiveness",
]
# Start of the reason of a site that failed with an error
FAILED_PREFIX = "failed: "


def failed_reason(error):
    """Drop reason of a site that failed with error."""
    return f"{FAILED_PREFIX}{error}"


def read_dropped(path=DROPPED_SITES):
    """The dropped sites table (empty if none recorded yet)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=COLUMNS)


def load_dropped(steps=None, path=DROPPED_SITES):
    """{site_id: reason} of the sites dropped at steps (default: any step; empty if
    none recorded yet)."""
    dropped = read_dropped(path)
    if steps is not None:
        dropped = dropped[dropped["step"].isin(steps)]
    return dict(zip(dropped["site_id"].astype(int), dropped["reason"]))


def load_dropped_before(step, path=DROPPED_SITES):
    """{site_id: reason} of the sites dropped at steps before step."""
    return load_dropped(STEPS[: STEPS.index(step)], path)


def reason_kind(reason):
    """A drop reason without its site's own count or error message."""
    if reason.startswith(FAILED_PREFIX):
        return "failed with an error"
    return re.sub(r" \(\d+\)$", "", reason)


def report_dropped(step, path=DROPPED_SITES, sites_path=SITE_GEOMS):
    """Print how many of the group's sites were dropped before step and at step (by
    reason), and how many remain."""
    dropped = read_dropped(path)
    n_cleaned = len(pd.read_parquet(sites_path, columns=["site_id"]))
    n_total = n_cleaned + int((dropped["step"] == "clean_sites").sum())
    earlier = dropped[dropped["step"].isin(STEPS[: STEPS.index(step)])]
    here = dropped[dropped["step"] == step]
    n_remain = n_total - len(earlier) - len(here)
    by_step = ", ".join(
        f"{s}: {n}" for s in STEPS if (n := int((earlier["step"] == s).sum()))
    )
    lines = [
        f"\nSites after {step} (dropped sites in {path}):",
        f"  {n_total:6d}  in the group",
        f"  {-len(earlier):6d}  dropped at earlier steps"
        + (f" ({by_step})" if by_step else ""),
        f"  {-len(here):6d}  dropped at {step}",
    ]
    for kind, n in here["reason"].map(reason_kind).value_counts().items():
        lines.append(f"  {'':6s}    {n:6d}  {kind}")
    share = f" ({n_remain / n_total:.0%} of {n_total})" if n_total else ""
    lines.append(f"  {n_remain:6d}  remain{share}")
    print("\n".join(lines))


def record_dropped(step, reasons, path=DROPPED_SITES, sites_path=SITE_GEOMS):
    """Replace the sites dropped at step with reasons ({site_id: reason}), clear those
    dropped at later steps, and print how many remain (report_dropped)."""
    dropped = read_dropped(path)
    later = STEPS[STEPS.index(step) + 1 :]
    stale = dropped["step"].isin(later)
    if stale.any():
        print(
            f"Cleared {stale.sum()} site(s) dropped at {', '.join(later)}; "
            "re-run those steps to update them"
        )
    new = pd.DataFrame(
        {
            "site_id": [int(site_id) for site_id in reasons],
            "step": step,
            "reason": list(reasons.values()),
        }
    )
    keep = ~dropped["step"].isin([step, *later])
    dropped = pd.concat([dropped[keep], new], ignore_index=True)
    dropped.reindex(columns=COLUMNS).to_csv(path, index=False)
    report_dropped(step, path, sites_path)
