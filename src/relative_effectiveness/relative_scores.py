"""
PA-level relative effectiveness from matched treatment and control cell scores.

Per-cell scores can be pulled two ways:
  - `scores_to_df` for an interactive fetch, which is subject to Earth Engine's
    ~5 minute interactive computation limit
  - `export_scores_to_gcs` + `load_scores` for a batch export, which has no such
    limit and is the reliable path for sites with many matched cells
"""

import time
from pathlib import Path

import ee
import geemap
import pandas as pd

from utils.variables import GCS_BUCKET, RELATIVE_SCORES_PREFIX

SCORE_COLS = ["extent_score", "intactness_score", "condition_score", "loss_score"]
DIFF_COLS = [f"{col}_diff" for col in SCORE_COLS]
EXPORT_COLS = ["cell_ID"] + SCORE_COLS

__all__ = [
    "SCORE_COLS",
    "DIFF_COLS",
    "EXPORT_COLS",
    "load_match_table",
    "scores_to_df",
    "scores_gcs_path",
    "export_scores_to_gcs",
    "wait_for_export",
    "load_scores",
    "build_match_df_with_score_diffs",
    "aggregate_pa_relative_scores",
]


def load_match_table(site_id: int, data_dir: str | Path = "data") -> pd.DataFrame:
    """Load the PSM match table for a site."""
    path = Path(data_dir) / f"mdm/match_table_mdm_{site_id}.parquet"
    mt = pd.read_parquet(path)
    mt["treat_cell_id"] = mt["treat_cell_id"].astype(int)
    mt["control_cell_id"] = mt["control_cell_id"].astype(int)
    return mt


def scores_to_df(scored) -> pd.DataFrame:
    """Fetch per-cell scores interactively.

    Uses `ee_to_df` rather than `ee_to_gdf`: the latter evaluates the scored
    graph twice (once for a CRS lookup) and returns geometries we don't use.
    Sites whose score graph takes over ~5 minutes will still time out here —
    use the `export_scores_to_gcs` / `load_scores` batch path for those.
    """
    return geemap.ee_to_df(scored, columns=EXPORT_COLS)


def scores_gcs_path(
    site_id: int,
    bucket: str = GCS_BUCKET,
    prefix: str = RELATIVE_SCORES_PREFIX,
) -> str:
    """GCS URI of the exported per-cell scores CSV for a site."""
    return f"gs://{bucket}/{prefix}scores_{site_id}.csv"


def export_scores_to_gcs(
    scored,
    site_id: int,
    bucket: str = GCS_BUCKET,
    prefix: str = RELATIVE_SCORES_PREFIX,
    start: bool = True,
) -> ee.batch.Task:
    """Export per-cell scores to GCS as a batch task.

    Batch tasks are not bound by the interactive computation time limit, so this
    is the reliable way to evaluate the score graph for large sites. Geometries
    are dropped by listing only score columns in `selectors`.

    Returns the task, started unless `start=False`.
    """
    task = ee.batch.Export.table.toCloudStorage(
        collection=scored,
        description=f"relative_scores_{site_id}",
        bucket=bucket,
        fileNamePrefix=f"{prefix}scores_{site_id}",
        fileFormat="CSV",
        selectors=EXPORT_COLS,
    )
    if start:
        task.start()
    return task


def wait_for_export(
    task: ee.batch.Task,
    poll_interval: int = 30,
    max_wait_minutes: float = 60,
    verbose: bool = True,
) -> str:
    """Block until an export task reaches a terminal state or the wait elapses.

    Returns the final EE task state. Raises if the task failed or was cancelled.
    """
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}
    deadline = time.time() + max_wait_minutes * 60

    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {task.id} — {state}")
        if state in terminal:
            break
        if time.time() >= deadline:
            raise TimeoutError(
                f"Task {task.id} still {state} after {max_wait_minutes} minutes; "
                "it may still finish — check the EE Tasks tab and then call load_scores."
            )
        time.sleep(poll_interval)

    if state != "COMPLETED":
        raise RuntimeError(
            f"Task {task.id} ended as {state}: {status.get('error_message')}"
        )
    return state


def load_scores(
    site_id: int,
    bucket: str = GCS_BUCKET,
    prefix: str = RELATIVE_SCORES_PREFIX,
) -> pd.DataFrame:
    """Read per-cell scores exported by `export_scores_to_gcs`."""
    return pd.read_csv(scores_gcs_path(site_id, bucket, prefix))


def _prepare_scores(scores) -> pd.DataFrame:
    """Normalize scores given either as a scored FeatureCollection or a DataFrame."""
    df = scores if isinstance(scores, pd.DataFrame) else scores_to_df(scores)
    df = df[EXPORT_COLS].drop_duplicates("cell_ID").copy()
    df["cell_ID"] = df["cell_ID"].astype(int)
    return df


def build_match_df_with_score_diffs(scores, match_table: pd.DataFrame) -> pd.DataFrame:
    """Join per-cell scores to the match table and compute pairwise treatment–control differences.

    `scores` may be a scored ee.FeatureCollection (fetched interactively) or a
    DataFrame of scores already loaded from a batch export.
    """
    scores = _prepare_scores(scores)

    treat_renamed = {"cell_ID": "treat_cell_id", **{c: f"treat_{c}" for c in SCORE_COLS}}
    ctrl_renamed = {"cell_ID": "control_cell_id", **{c: f"control_{c}" for c in SCORE_COLS}}

    match_df = match_table.merge(
        scores.rename(columns=treat_renamed), on="treat_cell_id", how="left"
    ).merge(scores.rename(columns=ctrl_renamed), on="control_cell_id", how="left")

    for col in SCORE_COLS:
        match_df[f"{col}_diff"] = match_df[f"treat_{col}"] - match_df[f"control_{col}"]

    return match_df


def aggregate_pa_relative_scores(match_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Average score differences in two steps to avoid weighting cells with more matches.

    Returns per-treatment-cell mean differences and PA-level relative scores.
    """
    treat_means = match_df.groupby("treat_cell_id")[DIFF_COLS].mean()

    pa_scores = {
        "extent": treat_means["extent_score_diff"].mean(),
        "intactness": treat_means["intactness_score_diff"].mean(),
        "condition": treat_means["condition_score_diff"].mean(),
        "loss": treat_means["loss_score_diff"].mean(),
    }

    return treat_means, pa_scores
