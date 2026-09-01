"""
PA-level relative effectiveness from matched treatment and control cell scores.
"""

from pathlib import Path

import geemap
import pandas as pd

SCORE_COLS = ["extent_score", "intactness_score", "condition_score", "loss_score"]
DIFF_COLS = [f"{col}_diff" for col in SCORE_COLS]

__all__ = [
    "SCORE_COLS",
    "DIFF_COLS",
    "load_match_table",
    "build_match_df_with_score_diffs",
    "aggregate_pa_relative_scores",
]


def load_match_table(site_id: int, match_method: str = "mdm", data_dir: str | Path = "data") -> pd.DataFrame:
    f"""Load the MDM/PSM match table for a site."""
    path = Path(data_dir) / match_method / f"match_table_{match_method}_{site_id}.parquet"
    mt = pd.read_parquet(path)
    mt["treat_cell_id"] = mt["treat_cell_id"].astype(int)
    mt["control_cell_id"] = mt["control_cell_id"].astype(int)
    return mt


def build_match_df_with_score_diffs(scored, match_table: pd.DataFrame) -> pd.DataFrame:
    """Join per-cell scores to the match table and compute pairwise treatment–control differences."""
    score_gdf = geemap.ee_to_gdf(scored)
    scores = score_gdf[["cell_ID"] + SCORE_COLS].drop_duplicates("cell_ID").copy()
    scores["cell_ID"] = scores["cell_ID"].astype(int)

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
