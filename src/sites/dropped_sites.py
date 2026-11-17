"""
Record and read sites dropped from the analysis (DROPPED_SITES, one row per site with
the step that dropped it and why). A step drops sites that fail one of its rules (e.g.
too few treatment cells) and sites that failed with an error (reason "failed: <error>").

Each step skips the sites dropped at earlier steps. Recording a step clears the records
of later steps, which are out of date until those steps re-run.
"""

import pandas as pd

from utils.variables import DROPPED_SITES

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


def load_dropped(steps=None, path=DROPPED_SITES):
    """{site_id: reason} of the sites dropped at steps (default: any step; empty if
    none recorded yet)."""
    try:
        dropped = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    if steps is not None:
        dropped = dropped[dropped["step"].isin(steps)]
    return dict(zip(dropped["site_id"].astype(int), dropped["reason"]))


def load_dropped_before(step, path=DROPPED_SITES):
    """{site_id: reason} of the sites dropped at steps before step."""
    return load_dropped(STEPS[: STEPS.index(step)], path)


def record_dropped(step, reasons, path=DROPPED_SITES):
    """Replace the sites dropped at step with reasons ({site_id: reason}), and clear
    those dropped at later steps."""
    try:
        dropped = pd.read_csv(path)
    except FileNotFoundError:
        dropped = pd.DataFrame(columns=COLUMNS)
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
    print(f"{len(reasons)} site(s) dropped at {step}; recorded in {path}")
