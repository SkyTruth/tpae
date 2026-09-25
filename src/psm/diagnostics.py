"""Covariate balance diagnostics for matched treatment and control cells."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.variables import COVARIATES

DIAGNOSTIC_COLUMNS = [
    "site_id",
    "match_coverage",
    "avg_matches_per_treat",
    "avg_control_reuse",
    "avg_extrapolation",
    "avg_abs_smd_before",
    "avg_abs_smd_after",
    "n_covariates_balanced",
    "avg_abs_smd_improvement",
    "n_covariates_improved",
]


def calc_match_coverage(match_df, treat_df):
    """Calculate match coverage as the ratio of matched treatment cells to total treatment cells."""
    return (
        match_df["treat_cell_id"].nunique() / len(treat_df) if len(treat_df) > 0 else 0
    )


def calc_avg_matches_per_treat(match_df):
    """Calculate average matches per treatment cell."""
    return match_df.groupby("treat_cell_id").size().mean()


def calc_control_reuse(match_df):
    """Calculate average number of times a control cell is reused."""
    return match_df.groupby("control_cell_id").size().mean()


def calc_extrapolation(t_vals, c_vals):
    """Calculate percentage of treatment cells that fall outside the range of control cells.
    If treatment cells have covariate values outside the range of available control cells,
    they cannot be properly matched and balanced by any method."""
    c_min = c_vals.min()
    c_max = c_vals.max()
    n_below = (t_vals < c_min).sum()
    n_above = (t_vals > c_max).sum()
    n_total = n_below + n_above
    pct = n_total / len(t_vals)
    return pct


def calc_pooled_sd(t_vals, c_vals):
    """Pooled standard deviation formula."""
    var_t, var_c = t_vals.var(), c_vals.var()  # sample variances
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd == 0:
        return 0.0
    return pooled_sd


def calc_smd(t_vals, c_vals, pooled_sd):
    """Standardized mean difference formula."""
    mean_t, mean_c = t_vals.mean(), c_vals.mean()  # sample means
    return (mean_t - mean_c) / pooled_sd


def balance_verdict(smd):
    """A standardized mean difference of 0.2 or less after matching indicates
    that a covariate is balanced (Feng et al. 2022)."""
    if abs(smd) <= 0.2:
        return 1
    else:
        return 0


def calc_improvement(smd_before, smd_after):
    """Did matching decrease the SMD?"""
    return abs(smd_before) - abs(smd_after)


def evaluate_covariate_balance(match_df, cells_df):
    """Calculate balance diagnostics for each covariate."""
    # Before matching: full treatment and control pools
    unmatched_treat = cells_df[cells_df["protected"] == 1][COVARIATES]
    unmatched_control = cells_df[cells_df["protected"] == 0][COVARIATES]

    # After matching: matched treatment and control cells
    matched_treat = match_df.merge(
        cells_df[["cell_ID"] + COVARIATES],
        left_on="treat_cell_id",
        right_on="cell_ID",
    ).drop(columns="cell_ID")

    matched_control = match_df.merge(
        cells_df[["cell_ID"] + COVARIATES],
        left_on="control_cell_id",
        right_on="cell_ID",
    ).drop(columns="cell_ID")

    rows = []
    for covariate in COVARIATES:
        extrapolation = calc_extrapolation(
            unmatched_treat[covariate], unmatched_control[covariate]
        )
        pooled_sd = calc_pooled_sd(
            unmatched_treat[covariate], unmatched_control[covariate]
        )
        smd_before = calc_smd(
            unmatched_treat[covariate], unmatched_control[covariate], pooled_sd
        )
        smd_after = calc_smd(
            matched_treat[covariate], matched_control[covariate], pooled_sd
        )
        balanced = balance_verdict(smd_after)
        improvement = calc_improvement(smd_before, smd_after)
        rows.append(
            {
                "covariate": covariate,
                "extrapolation": extrapolation,
                "smd_before": smd_before,
                "smd_after": smd_after,
                "balanced": balanced,
                "improvement": improvement,
            }
        )

    covariate_results = pd.DataFrame(rows)
    return covariate_results


def evaluate_overall_balance(covariate_results):
    """Evaluate overall site balance across all covariates."""
    avg_extrapolation = covariate_results["extrapolation"].mean()
    avg_abs_SDM_before = covariate_results["smd_before"].abs().mean()
    avg_abs_SDM_after = covariate_results["smd_after"].abs().mean()
    n_covariates_balanced = covariate_results["balanced"].sum()
    avg_abs_SDM_improvement = covariate_results["improvement"].mean()
    n_covariates_improved = (covariate_results["improvement"] > 0).sum()
    return (
        avg_extrapolation,
        avg_abs_SDM_before,
        avg_abs_SDM_after,
        n_covariates_balanced,
        avg_abs_SDM_improvement,
        n_covariates_improved,
    )


def site_diagnostics_row(match_df, treat_df, cells_df, site_id):
    """Compile all site-level diagnostics for a site in a single row."""
    has_matches = (
        match_df is not None
        and len(match_df) > 0
        and "treat_cell_id" in match_df.columns
    )

    row = {col: np.nan for col in DIAGNOSTIC_COLUMNS}
    row["site_id"] = site_id
    row["match_coverage"] = (
        calc_match_coverage(match_df, treat_df) if has_matches else 0.0
    )
    row["n_covariates_balanced"] = 0
    row["n_covariates_improved"] = 0

    if not has_matches:
        return row

    covariate_results = evaluate_covariate_balance(match_df, cells_df)
    (
        avg_extrapolation,
        avg_abs_smd_before,
        avg_abs_smd_after,
        n_covariates_balanced,
        avg_abs_smd_improvement,
        n_covariates_improved,
    ) = evaluate_overall_balance(covariate_results)

    row.update(
        {
            "avg_matches_per_treat": calc_avg_matches_per_treat(match_df),
            "avg_control_reuse": calc_control_reuse(match_df),
            "avg_extrapolation": avg_extrapolation,
            "avg_abs_smd_before": avg_abs_smd_before,
            "avg_abs_smd_after": avg_abs_smd_after,
            "n_covariates_balanced": n_covariates_balanced,
            "avg_abs_smd_improvement": avg_abs_smd_improvement,
            "n_covariates_improved": n_covariates_improved,
        }
    )
    return row


def save_experiment_diagnostics(site_rows, output_path):
    """Write one-row-per-site experiment diagnostics to a single CSV (local or gs:// path)."""
    if "://" not in output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(site_rows).reindex(columns=DIAGNOSTIC_COLUMNS)
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} site rows to {output_path}")
    return results_df
