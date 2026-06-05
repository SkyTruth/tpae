"""
Propensity-score prediction and nearest-neighbor matching for PA cells.
"""

from pathlib import Path

import ee
import geemap
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from psm.predict import load_propensity_artifacts, predict_propensity
from utils.variables import CALIPER, N_NEIGHBORS


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


def match_treatment_control(cells_df):
    """Match each treatment cell to control cells by propensity score within strata."""
    treat_df = cells_df[cells_df["protected"] == 1].copy().reset_index(drop=True)
    control_df = cells_df[cells_df["protected"] == 0].copy().reset_index(drop=True)

    matches = []

    for (country, ecoregion), treat_sub in treat_df.groupby(["country", "ecoregion"]):
        control_country = control_df[control_df["country"] == country]

        if len(control_country) == 0:
            print(
                f"  ({country}, ecoregion {ecoregion}): no controls in country, "
                f"skipping {len(treat_sub)} treatment cells"
            )
            continue

        control_sub = control_country[control_country["ecoregion"] == ecoregion]

        if len(control_sub) == 0:
            biome = treat_sub["biome"].iloc[0]
            control_sub = control_country[control_country["biome"] == biome]
            fallback = "biome"
            print(
                f"  ({country}, ecoregion {ecoregion}): no within-ecoregion controls, "
                f"falling back to biome {biome} ({len(control_sub)} controls)"
            )
        else:
            fallback = None

        if len(control_sub) == 0:
            print(
                f"  ({country}, ecoregion {ecoregion}): no controls at any fallback level, "
                f"skipping {len(treat_sub)} treatment cells"
            )
            continue

        n_neighbors = min(N_NEIGHBORS, len(control_sub))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(control_sub[["propensity_score"]].values)

        distances, indices = nn.kneighbors(treat_sub[["propensity_score"]].values)

        for i, treat_row in enumerate(treat_sub.itertuples()):
            for rank, (dist, j) in enumerate(zip(distances[i], indices[i]), start=1):
                if dist <= CALIPER:
                    control_row = control_sub.iloc[j]
                    matches.append(
                        {
                            "treat_cell_id": treat_row.cell_ID,
                            "control_cell_id": control_row["cell_ID"],
                            "treat_score": treat_row.propensity_score,
                            "control_score": control_row["propensity_score"],
                            "ps_distance": float(dist),
                            "match_rank": rank,
                            "match_country": country,
                            "match_ecoregion": ecoregion,
                            "match_fallback": fallback,
                        }
                    )

    match_df = pd.DataFrame(matches).sort_values("treat_cell_id").reset_index(drop=True)

    print(f"\nResults:")
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

    print(f"\nFirst 10 matches:")
    print(match_df.head(10) if len(match_df) > 0 else "(no matches)")

    return match_df, treat_df, control_df


def filter_matched_grids(grid_fc, match_df):
    """Filter the grid FeatureCollection to cells that appear in the match table."""
    valid_ids = pd.concat([match_df["treat_cell_id"], match_df["control_cell_id"]]).unique()
    valid_ids = ee.List(valid_ids.astype(int).tolist())
    return grid_fc.filter(ee.Filter.inList("cell_ID", valid_ids))


def save_matching_outputs(matched_grids, match_df, pa_id, data_dir="data"):
    """Write matched grids and match pairs to parquet."""
    matched_grids_gdf = geemap.ee_to_gdf(matched_grids)
    matched_grids_gdf.to_parquet(f"{data_dir}/matched_grids_{pa_id}.parquet")
    match_df.to_parquet(f"{data_dir}/match_table_{pa_id}.parquet", index=False)
