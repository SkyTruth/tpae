"""
Record and read sites dropped from the analysis for too few treatment cells
(DROPPED_SITES, one row per site with the step that dropped it and why).
"""

import pandas as pd

from utils.variables import DROPPED_SITES

COLUMNS = ["site_id", "step", "reason"]


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
    """Replace the sites dropped at step with reasons ({site_id: reason})."""
    try:
        dropped = pd.read_csv(path)
    except FileNotFoundError:
        dropped = pd.DataFrame(columns=COLUMNS)
    new = pd.DataFrame(
        {
            "site_id": [int(site_id) for site_id in reasons],
            "step": step,
            "reason": list(reasons.values()),
        }
    )
    dropped = pd.concat([dropped[dropped["step"] != step], new], ignore_index=True)
    dropped.reindex(columns=COLUMNS).to_csv(path, index=False)
    print(f"{len(reasons)} site(s) dropped at {step}; recorded in {path}")
