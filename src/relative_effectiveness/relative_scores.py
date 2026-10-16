"""
PA-level relative effectiveness from matched treatment and control cell scores.
"""

import geemap
import numpy as np
import pandas as pd

from psm.diagnostics import calc_match_weights
from utils.variables import COVARIATES, REG_ADJ_MIN_CONTROLS_PER_COVARIATE

SCORE_COLS = ["extent_score", "intactness_score", "condition_score", "loss_score"]
DIFF_COLS = [f"{col}_diff" for col in SCORE_COLS]
ADJ_DIFF_COLS = [f"{col}_diff_adj" for col in SCORE_COLS]
PA_SCORE_NAMES = {
    "extent_score": "extent",
    "intactness_score": "intactness",
    "condition_score": "condition",
    "loss_score": "loss",
}

__all__ = [
    "SCORE_COLS",
    "DIFF_COLS",
    "ADJ_DIFF_COLS",
    "build_match_df_with_score_diffs",
    "regression_adjust_score_diffs",
    "aggregate_pa_relative_scores",
]


def build_match_df_with_score_diffs(scored, match_table: pd.DataFrame) -> pd.DataFrame:
    """Join per-cell scores to the match table and compute pairwise treatment–control differences."""
    score_gdf = geemap.ee_to_gdf(scored)
    scores = score_gdf[["cell_ID"] + SCORE_COLS].drop_duplicates("cell_ID").copy()
    scores["cell_ID"] = scores["cell_ID"].astype(int)

    treat_renamed = {
        "cell_ID": "treat_cell_id",
        **{c: f"treat_{c}" for c in SCORE_COLS},
    }
    ctrl_renamed = {
        "cell_ID": "control_cell_id",
        **{c: f"control_{c}" for c in SCORE_COLS},
    }

    match_df = match_table.merge(
        scores.rename(columns=treat_renamed), on="treat_cell_id", how="left"
    ).merge(scores.rename(columns=ctrl_renamed), on="control_cell_id", how="left")

    for col in SCORE_COLS:
        match_df[f"{col}_diff"] = match_df[f"treat_{col}"] - match_df[f"control_{col}"]

    return match_df


def _fit_control_outcome_slopes(X_control, y_control, weights):
    """Weighted least-squares slopes of an outcome on covariates, over matched controls.

    Returns one slope per covariate (0 for covariates with no variation among the
    controls), or None if the outcome itself doesn't vary.
    """
    if np.ptp(y_control) == 0:
        return None
    varies = np.ptp(X_control, axis=0) > 0
    design = np.column_stack([np.ones(len(X_control)), X_control[:, varies]])
    sqrt_w = np.sqrt(weights)[:, None]
    coefs, *_ = np.linalg.lstsq(design * sqrt_w, y_control * sqrt_w[:, 0], rcond=None)
    slopes = np.zeros(X_control.shape[1])
    slopes[varies] = coefs[1:]
    return slopes


def regression_adjust_score_diffs(
    match_df: pd.DataFrame,
    cell_covariates: pd.DataFrame,
    covariates=COVARIATES,
    min_controls_per_covariate=REG_ADJ_MIN_CONTROLS_PER_COVARIATE,
) -> pd.DataFrame:
    """Add regression-adjusted score differences ({score}_diff_adj) to match_df.

    For each score, a linear regression model is fit on the matched control cells
    (weighted by how often each is used, with 1/k pair weights).
    Each pair's difference in effectiveness is then corrected for the part
    explained by its covariate gap:
        diff_adj = diff - slopes · (x_treat - x_control)

    cell_covariates: one row per cell with cell_ID and the covariate columns (e.g. the
    matched grids). With fewer than min_controls_per_covariate matched controls per
    covariate, no model is fit and diff_adj equals diff. Whether the adjustment was
    applied is stored in match_df.attrs["regression_adjusted"].
    """
    match_df = match_df.copy()
    covs = cell_covariates.drop_duplicates("cell_ID").set_index("cell_ID")[covariates]
    X_treat = covs.loc[match_df["treat_cell_id"]].to_numpy(dtype=float)
    X_ctrl = covs.loc[match_df["control_cell_id"]].to_numpy(dtype=float)

    # Each control's weight is the sum of its pairs' 1/k weights
    pair_weights = pd.Series(
        calc_match_weights(match_df).to_numpy(), index=match_df["control_cell_id"]
    )
    control_weights = pair_weights.groupby(level=0).sum()
    n_controls = len(control_weights)
    adjusted = n_controls >= min_controls_per_covariate * len(covariates)

    for col in SCORE_COLS:
        diff = match_df[f"{col}_diff"].to_numpy(dtype=float)
        if not adjusted:
            match_df[f"{col}_diff_adj"] = diff
            continue
        controls = (
            match_df[["control_cell_id", f"control_{col}"]]
            .drop_duplicates("control_cell_id")
            .set_index("control_cell_id")[f"control_{col}"]
            .dropna()
        )
        slopes = _fit_control_outcome_slopes(
            covs.loc[controls.index].to_numpy(dtype=float),
            controls.to_numpy(dtype=float),
            control_weights.loc[controls.index].to_numpy(),
        )
        correction = 0.0 if slopes is None else (X_treat - X_ctrl) @ slopes
        match_df[f"{col}_diff_adj"] = diff - correction

    match_df.attrs["regression_adjusted"] = adjusted
    return match_df


def aggregate_pa_relative_scores(
    match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Average score differences in two steps to avoid weighting cells with more matches.

    Returns per-treatment-cell mean differences and PA-level relative scores. If
    match_df has regression-adjusted differences (see regression_adjust_score_diffs),
    adjusted PA scores are added as "{name}_adj" (e.g. "loss_adj").
    """
    diff_cols = DIFF_COLS + [col for col in ADJ_DIFF_COLS if col in match_df]
    treat_means = match_df.groupby("treat_cell_id")[diff_cols].mean()

    pa_scores = {}
    for col, name in PA_SCORE_NAMES.items():
        pa_scores[name] = treat_means[f"{col}_diff"].mean()
        if f"{col}_diff_adj" in treat_means:
            pa_scores[f"{name}_adj"] = treat_means[f"{col}_diff_adj"].mean()

    return treat_means, pa_scores
