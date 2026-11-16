"""
Record and read sites dropped from the analysis for too few treatment cells
(DROPPED_SITES, one row per site with the step that dropped it and why).

Recording a step clears the records of later steps, which are out of date until those
steps re-run.
"""

import pandas as pd

from utils.variables import DROPPED_SITES

COLUMNS = ["site_id", "step", "reason"]
# Steps that drop sites, in pipeline order
STEPS = ["treatment_cells", "matching"]


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
