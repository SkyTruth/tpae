"""
Functions for matching treatment and control cells using Propensity Score Matching (PSM).
"""

from pathlib import Path

import pandas as pd

from archive.PSM_predict import load_propensity_artifacts, predict_propensity
from matching.match_cells import _print_waterfall_pools, waterfall_in_caliper_neighbors
from utils.variables import CALIPER_PSM, N_NEIGHBORS_PSM


def add_propensity_scores(cells_df, models_dir="models"):
    """Load the latest saved propensity model and score each cell."""
    model_files = sorted(Path(models_dir).glob("propensity_model_*.pkl"))
    artifacts = load_propensity_artifacts(model_files[-1])
    print(f"Loaded {model_files[-1].name}")

    cells_df = cells_df.copy()
    cells_df["propensity_score"] = predict_propensity(cells_df, artifacts)

    print(f"\nCells: {len(cells_df)}")
    print(f"Treatment (protected=1): {(cells_df['protected'] == 1).sum()}")
    print(f"Control (protected=0): {(cells_df['protected'] == 0).sum()}")
    # print(f"\nPropensity score distribution:")
    # print(cells_df["propensity_score"].describe())
    # print(f"\nFirst 5 rows:")
    print(cells_df.head())

    return cells_df


def match_treatment_control_psm(cells_df):
    """Match each treatment cell to control cells by Propensity Score Matching (PSM)."""
    treat_df = cells_df[cells_df["protected"] == 1].copy().reset_index(drop=True)
    control_df = cells_df[cells_df["protected"] == 0].copy().reset_index(drop=True)
    control_by_id = control_df.set_index("cell_ID")

    pending, _, _ = waterfall_in_caliper_neighbors(
        treat_df,
        control_df,
        ["propensity_score"],
        CALIPER_PSM,
        {"metric": "euclidean"},
    )
    _print_waterfall_pools(pending, len(treat_df))

    matches = []
    for _, treat_row, fallback, candidates in pending:
        k = min(N_NEIGHBORS_PSM, len(candidates))
        for rank, (dist, control_id) in enumerate(candidates[:k], start=1):
            control_row = control_by_id.loc[control_id]
            matches.append(
                {
                    "treat_cell_id": treat_row.cell_ID,
                    "control_cell_id": control_id,
                    "treat_score": treat_row.propensity_score,
                    "control_score": control_row["propensity_score"],
                    "ps_distance": float(dist),
                    "match_rank": rank,
                    "match_country": treat_row.country,
                    "match_ecoregion": treat_row.ecoregion,
                    "match_fallback": fallback,
                }
            )

    if matches:
        match_df = (
            pd.DataFrame(matches).sort_values("treat_cell_id").reset_index(drop=True)
        )
    else:
        match_df = pd.DataFrame(
            columns=[
                "treat_cell_id",
                "control_cell_id",
                "treat_score",
                "control_score",
                "ps_distance",
                "match_rank",
                "match_country",
                "match_ecoregion",
                "match_fallback",
            ]
        )

    print("\nResults:")
    print(f"  Total matched pairs: {len(match_df)}")
    print(f"  Unique treatment cells matched: {match_df['treat_cell_id'].nunique()}")
    print(f"  Unique control cells used: {match_df['control_cell_id'].nunique()}")

    unmatched_treat = set(treat_df["cell_ID"]) - set(match_df["treat_cell_id"])
    print(f"  Treatment cells with no match: {len(unmatched_treat)}")

    match_coverage = match_df["treat_cell_id"].nunique() / len(treat_df)
    print(f"  Match coverage: {match_coverage:.1%}")

    if len(match_df) > 0:
        avg_matches = match_df.groupby("treat_cell_id").size().mean()
        print(f"  Avg matches per matched treatment cell: {avg_matches:.2f}")

        control_reuse = match_df.groupby("control_cell_id").size()
        print(
            f"  Control reuse: min={control_reuse.min()}, "
            f"max={control_reuse.max()}, "
            f"mean={control_reuse.mean():.1f}"
        )

    print("\nFirst 10 matches:")
    print(match_df.head(10) if len(match_df) > 0 else "(no matches)")

    return match_df, treat_df, control_df
