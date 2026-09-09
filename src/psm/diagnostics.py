"""Covariate balance diagnostics for matched treatment and control cells."""

import numpy as np
import pandas as pd

from utils.variables import COVARIATES


def compute_pair_smd(matched_t_vals, matched_c_vals, full_t_vals, full_c_vals):
    """
    Pair-level standardized mean difference.

    Returns the mean absolute pair distance (in pooled-SD units) plus
    the 90th percentile, which surfaces worst-pair imbalance.

    Pooled SD is computed on the FULL (unmatched) treatment and control pools
    to anchor the metric to the original covariate scale.

    Parameters
    ----------
    matched_t_vals, matched_c_vals : pandas Series, equal length
        Covariate values for matched treatment and control cells (in pair order).
    full_t_vals, full_c_vals : pandas Series
        Covariate values for the full unmatched treatment and control pools.
        Used to compute the pooled SD that anchors the metric.

    Returns
    -------
    (mean_abs_smd, p90_abs_smd, signed_smd) : tuple of floats
    """
    var_full_t = full_t_vals.var()
    var_full_c = full_c_vals.var()
    pooled_sd = np.sqrt((var_full_t + var_full_c) / 2)
    if pooled_sd == 0:
        return 0.0, 0.0, 0.0

    pair_diffs = (matched_t_vals.values - matched_c_vals.values) / pooled_sd
    return (
        np.abs(pair_diffs).mean(),
        np.percentile(np.abs(pair_diffs), 90),
        pair_diffs.mean(),  # signed mean for directional info
    )


def pair_smd_verdict(mean_abs):
    """Label pair-level mean |SMD| as excellent, acceptable, or imbalanced."""
    if mean_abs < 0.10:
        return "excellent"
    if mean_abs < 0.25:
        return "acceptable"
    return "IMBALANCED"


def pair_covariate_balance(match_df, cells_df, covariates=None):
    """
    Pair-level covariate balance for matched treatment/control cells.

    Prints the per-covariate table and returns a DataFrame with mean |SMD|,
    90th-percentile |SMD|, signed mean, and verdict.
    """
    if covariates is None:
        covariates = COVARIATES

    matched_treat = match_df.merge(
        cells_df[["cell_ID"] + list(covariates)],
        left_on="treat_cell_id",
        right_on="cell_ID",
    ).drop(columns="cell_ID")

    matched_control = match_df.merge(
        cells_df[["cell_ID"] + list(covariates)],
        left_on="control_cell_id",
        right_on="cell_ID",
    ).drop(columns="cell_ID")

    unmatched_treat = cells_df[cells_df["protected"] == 1]
    unmatched_control = cells_df[cells_df["protected"] == 0]

    results = []
    for col in covariates:
        mean_abs, p90, signed = compute_pair_smd(
            matched_treat[col],
            matched_control[col],
            unmatched_treat[col],
            unmatched_control[col],
        )
        results.append(
            {
                "covariate": col,
                "mean_abs_smd": mean_abs,
                "p90_abs_smd": p90,
                "signed_mean": signed,
                "verdict": pair_smd_verdict(mean_abs),
            }
        )

    results_df = pd.DataFrame(results)

    print("Pair-level balance check")
    print("Mean absolute pair SMD (lower = better individual match quality)")
    print("Threshold: < 0.25 = acceptable; < 0.10 = excellent")
    print("=" * 90)
    print(
        f"{'covariate':<20s} {'mean |smd|':>12s} {'p90 |smd|':>12s} "
        f"{'signed mean':>14s} {'verdict':>20s}"
    )
    print("-" * 90)
    for row in results:
        print(
            f"{row['covariate']:<20s} {row['mean_abs_smd']:>12.3f} "
            f"{row['p90_abs_smd']:>12.3f} {row['signed_mean']:>+14.3f} "
            f"{row['verdict']:>20s}"
        )

    return results_df
