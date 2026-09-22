"""Covariate balance diagnostics for matched treatment and control cells."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.variables import COVARIATES

ABS_SMD_AFTER_COLS = [f"abs_smd_after_{c}" for c in COVARIATES]

DIAGNOSTIC_COLUMNS = [
    "site_id",
    "match_coverage",
    "frac_treat_0_neighbors",
    "frac_treat_1_neighbor",
    "frac_treat_2plus_neighbors",
    "control_supply_ratio",
    "avg_matches_per_treat",
    "avg_control_reuse",
    "avg_extrapolation",
    "avg_abs_smd_before",
    "avg_abs_smd_after",
    *ABS_SMD_AFTER_COLS,
    "n_covariates_balanced",
    "avg_abs_smd_improvement",
    "n_covariates_improved",
]


def calc_match_coverage(match_df, treat_df):
    """Calculate match coverage as the ratio of matched treatment cells to total treatment cells."""
    return (
        match_df["treat_cell_id"].nunique() / len(treat_df) if len(treat_df) > 0 else 0
    )


def calc_control_supply_ratio(n_in_caliper_controls, n_treat, k, cap):
    """(in-caliper controls × reuse cap) / (n_treat × k).

    Values < 1 mean there are not enough reusable in-caliper controls to give
    every treatment cell k matches, so the control pool needs to be expanded.
    """
    demand = n_treat * k
    if demand == 0:
        return np.nan
    return (n_in_caliper_controls * cap) / demand


def calc_neighbor_count_shares(n_candidates_by_treat):
    """Share of treatment cells with 0, 1, or 2+ in-caliper control neighbors."""
    if not n_candidates_by_treat:
        return np.nan, np.nan, np.nan
    counts = np.fromiter(n_candidates_by_treat.values(), dtype=float)
    return (
        float((counts == 0).mean()),
        float((counts == 1).mean()),
        float((counts >= 2).mean()),
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
    """Standardized mean difference formula.

    If a covariate has no variation (pooled SD is 0) and the two groups have the
    same mean, treat SMD as 0 (balanced). If the means differ, SMD is undefined.
    """
    mean_t, mean_c = float(t_vals.mean()), float(c_vals.mean())
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return 0.0 if np.isclose(mean_t, mean_c) else np.nan
    return (mean_t - mean_c) / pooled_sd


def balance_verdict(smd):
    """A standardized mean difference of 0.2 or less after matching indicates
    that a covariate is balanced (Feng et al. 2022). Constant covariates with
    equal means are treated as balanced (SMD = 0)."""
    if pd.isna(smd):
        return 0
    if abs(smd) <= 0.2:
        return 1
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

    if match_df is not None:
        n_in_caliper = match_df.attrs.get("n_in_caliper_controls")
        cap = match_df.attrs.get("reuse_cap")
        k = match_df.attrs.get("n_neighbors")
        if n_in_caliper is not None and cap is not None and k is not None:
            row["control_supply_ratio"] = calc_control_supply_ratio(
                n_in_caliper, len(treat_df), k, cap
            )
        n_candidates = match_df.attrs.get("n_candidates_by_treat")
        if n_candidates is not None:
            frac0, frac1, frac2 = calc_neighbor_count_shares(n_candidates)
            row["frac_treat_0_neighbors"] = frac0
            row["frac_treat_1_neighbor"] = frac1
            row["frac_treat_2plus_neighbors"] = frac2

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
    abs_smds = covariate_results.set_index("covariate")["smd_after"].abs()
    for cov in COVARIATES:
        row[f"abs_smd_after_{cov}"] = abs_smds.get(cov, np.nan)
    return row


def save_experiment_diagnostics(site_rows, output_path):
    """Write one-row-per-site experiment diagnostics to a single CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(site_rows).reindex(columns=DIAGNOSTIC_COLUMNS)
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} site rows to {output_path}")
    return results_df
