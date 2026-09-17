"""
Functions for matching treatment and control cells, using PSM or MDM.
"""

from pathlib import Path

import ee
import geemap
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from psm.predict import load_propensity_artifacts, predict_propensity
from utils.variables import (
    CALIPER_MDM,
    CALIPER_PSM,
    COVARIATES,
    MAX_CONTROL_REUSE_CEILING,
    MAX_CONTROL_REUSE_FRAC,
    N_NEIGHBORS_MDM,
    N_NEIGHBORS_PSM,
)

"""
PSM matching functions
"""


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

        n_neighbors = min(N_NEIGHBORS_PSM, len(control_sub))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(control_sub[["propensity_score"]].values)

        distances, indices = nn.kneighbors(treat_sub[["propensity_score"]].values)

        for i, treat_row in enumerate(treat_sub.itertuples()):
            for rank, (dist, j) in enumerate(zip(distances[i], indices[i]), start=1):
                if dist <= CALIPER_PSM:
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


"""
MDM matching functions
"""


def fit_control_scaler_and_inv_cov(control_df, covariates):
    """StandardScaler and inverse covariance from control cells only (Stuart 2010)."""
    scaler = StandardScaler()
    X_control = scaler.fit_transform(control_df[covariates].values)
    cov_matrix = np.cov(X_control.T)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Warning: covariance matrix is singular; using pseudo-inverse")
        inv_cov = np.linalg.pinv(cov_matrix)
    return scaler, inv_cov


def calc_control_reuse_cap(
    n_treat,
    k,
    reuse_frac=MAX_CONTROL_REUSE_FRAC,
    reuse_ceiling=MAX_CONTROL_REUSE_CEILING,
):
    """Maximum number of times one control can be used."""
    return min(reuse_ceiling, max(1, int(np.ceil(reuse_frac * n_treat * k))))


def match_treatment_control_mdm(
    cells_df,
    covariates=None,
    caliper=CALIPER_MDM,
    n_neighbors=N_NEIGHBORS_MDM,
    reuse_frac=MAX_CONTROL_REUSE_FRAC,
    reuse_ceiling=MAX_CONTROL_REUSE_CEILING,
):
    """Match each treatment cell to control cells by Mahalanobis Distance Matching (MDM)."""
    if covariates is None:
        covariates = COVARIATES

    cells_df = cells_df.copy()
    control_only = cells_df[cells_df["protected"] == 0]
    scaler, inv_cov = fit_control_scaler_and_inv_cov(control_only, covariates)

    X_pa = scaler.transform(cells_df[covariates].values)
    for i, col in enumerate(covariates):
        cells_df[f"_scaled_{col}"] = X_pa[:, i]

    treat_df = cells_df[cells_df["protected"] == 1].copy().reset_index(drop=True)
    control_df = cells_df[cells_df["protected"] == 0].copy().reset_index(drop=True)

    cap = calc_control_reuse_cap(
        len(treat_df), n_neighbors, reuse_frac=reuse_frac, reuse_ceiling=reuse_ceiling
    )
    control_uses = {}

    print(f"Number of candidate treatment cells: {len(treat_df)}")
    print(f"Number of candidate control cells: {len(control_df)}")
    print(
        f"Control reuse cap: {cap} "
        f"(reuse_frac={reuse_frac}, reuse_ceiling={reuse_ceiling}, k={n_neighbors})"
    )

    scaled_cols = [f"_scaled_{c}" for c in covariates]
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

        k = min(n_neighbors, len(control_sub))
        nn = NearestNeighbors(
            metric="mahalanobis",
            metric_params={"VI": inv_cov},
        )
        nn.fit(control_sub[scaled_cols].values)
        distances, indices = nn.radius_neighbors(
            treat_sub[scaled_cols].values, radius=caliper
        )

        # Assign hardest-to-match treatment cells first (fewest in-caliper candidates)
        pending = []
        for i, treat_row in enumerate(treat_sub.itertuples()):
            candidates = sorted(zip(distances[i], indices[i]))
            pending.append((len(candidates), i, treat_row, candidates))
        pending.sort(key=lambda item: item[0])

        for _, _, treat_row, candidates in pending:
            n_matched = 0
            for dist, j in candidates:
                control_row = control_sub.iloc[j]
                control_id = control_row["cell_ID"]
                if control_uses.get(control_id, 0) >= cap:
                    continue
                control_uses[control_id] = control_uses.get(control_id, 0) + 1
                n_matched += 1
                matches.append(
                    {
                        "treat_cell_id": treat_row.cell_ID,
                        "control_cell_id": control_id,
                        "mahalanobis_distance": float(dist),
                        "match_rank": n_matched,
                        "match_country": country,
                        "match_ecoregion": ecoregion,
                        "match_fallback": fallback,
                    }
                )
                if n_matched >= k:
                    break

    if matches:
        match_df = (
            pd.DataFrame(matches).sort_values("treat_cell_id").reset_index(drop=True)
        )
    else:
        match_df = pd.DataFrame(
            columns=[
                "treat_cell_id",
                "control_cell_id",
                "mahalanobis_distance",
                "match_rank",
                "match_country",
                "match_ecoregion",
                "match_fallback",
            ]
        )

    print("\nResults:")
    print(f"  Treatment cells matched: {match_df['treat_cell_id'].nunique()}")
    print(f"  Unique control cells used: {match_df['control_cell_id'].nunique()}")
    print(f"  Total matched pairs: {len(match_df)}")
    if control_uses:
        print(
            f"  Control reuse: max={max(control_uses.values())} "
            f"(cap={cap}), mean={np.mean(list(control_uses.values())):.1f}"
        )

    return match_df, treat_df, control_df


"""
Save matched outputs
"""


def filter_matched_grids(grid_fc, match_df):
    """Filter the grid FeatureCollection to cells that appear in the match table."""
    valid_ids = pd.concat(
        [match_df["treat_cell_id"], match_df["control_cell_id"]]
    ).unique()
    valid_ids = ee.List(valid_ids.astype(int).tolist())
    return grid_fc.filter(ee.Filter.inList("cell_ID", valid_ids))


def save_matching_outputs(
    matched_grids, match_df, pa_id, match_method: str = "mdm", data_dir="data"
):
    """Write matched grids and match pairs to parquet."""
    matched_grids_gdf = geemap.ee_to_gdf(matched_grids)
    matched_grids_gdf.to_parquet(
        f"{data_dir}/{match_method}/matched_grids_{match_method}_{pa_id}.parquet"
    )
    match_df.to_parquet(
        f"{data_dir}/{match_method}/match_table_{match_method}_{pa_id}.parquet",
        index=False,
    )
