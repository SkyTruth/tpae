"""
Functions for matching treatment and control cells, using PSM or MDM.
"""

from collections import Counter
from pathlib import Path

import ee
import geemap
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from psm.diagnostics import calc_pooled_sd
from psm.predict import load_propensity_artifacts, predict_propensity
from utils.variables import (
    BALANCE_SMD_TARGET,
    CALIPER_MDM,
    CALIPER_PSM,
    COVARIATES,
    MAX_CONTROL_REUSE_CEILING,
    MAX_CONTROL_REUSE_FRAC,
    MATCHING_ALGORITHM,
    N_NEIGHBORS_MDM,
    OPTIMAL_MAX_CANDIDATES,
    OPTIMAL_NODE_LIMIT,
    OPTIMAL_TIME_LIMIT_S,
    N_NEIGHBORS_PSM,
    PER_COVARIATE_CALIPERS,
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


# Tightest pool first. A cell moves down only if it has 0 in-caliper neighbors.
WATERFALL_POOLS = (
    (None, ("country", "ecoregion")),
    ("biome", ("country", "biome")),
    ("ecoregion", ("ecoregion",)),
    ("biome_buffer", ("biome",)),
)
WATERFALL_POOL_LABELS = {
    None: "country × ecoregion",
    "biome": "country × biome",
    "ecoregion": "ecoregion in buffer",
    "biome_buffer": "biome in buffer",
}


def _as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def _control_pool(control_df, group_cols, key):
    mask = np.ones(len(control_df), dtype=bool)
    for col, val in zip(group_cols, key):
        mask &= control_df[col].values == val
    return control_df.iloc[np.flatnonzero(mask)]


def _in_caliper_candidates(
    treat_sub, control_sub, feature_cols, caliper, nn_kwargs, max_abs_diff=None
):
    """Map treat cell_ID → sorted (distance, control_cell_id) within caliper.

    max_abs_diff: optional {column: limit}; candidates must also have
    |treat - control| ≤ limit on each column.
    """
    if treat_sub.empty or control_sub.empty:
        return {}
    nn = NearestNeighbors(**nn_kwargs)
    nn.fit(control_sub[feature_cols].values)
    distances, indices = nn.radius_neighbors(
        treat_sub[feature_cols].values, radius=caliper
    )
    control_ids = control_sub["cell_ID"].values
    limits = [
        (treat_sub[col].values, control_sub[col].values, limit)
        for col, limit in (max_abs_diff or {}).items()
    ]
    out = {}
    for i, treat_row in enumerate(treat_sub.itertuples()):
        dist, idx = distances[i], indices[i]
        for treat_vals, control_vals, limit in limits:
            keep = np.abs(control_vals[idx] - treat_vals[i]) <= limit
            dist, idx = dist[keep], idx[keep]
        out[treat_row.cell_ID] = sorted(zip(dist, control_ids[idx]))
    return out


def waterfall_in_caliper_neighbors(
    treat_df, control_df, feature_cols, caliper, nn_kwargs, max_abs_diff=None
):
    """Lock each treatment cell to the tightest pool with ≥1 in-caliper neighbor.

    Cap-starved cells (neighbors exist, reuse cap takes them) stay in that pool.
    """
    n_candidates_by_treat = {cell_id: 0 for cell_id in treat_df["cell_ID"]}
    in_caliper_control_ids = set()
    pending = []
    remaining_ids = set(treat_df["cell_ID"])

    for fallback, group_cols in WATERFALL_POOLS:
        if not remaining_ids:
            break
        subset = treat_df[treat_df["cell_ID"].isin(remaining_ids)]
        for key, treat_sub in subset.groupby(list(group_cols), dropna=False):
            control_sub = _control_pool(control_df, group_cols, _as_tuple(key))
            cand_map = _in_caliper_candidates(
                treat_sub, control_sub, feature_cols, caliper, nn_kwargs, max_abs_diff
            )
            for treat_row in treat_sub.itertuples():
                candidates = cand_map.get(treat_row.cell_ID, [])
                if not candidates:
                    continue
                remaining_ids.discard(treat_row.cell_ID)
                n_candidates_by_treat[treat_row.cell_ID] = len(candidates)
                for _, control_id in candidates:
                    in_caliper_control_ids.add(control_id)
                pending.append((len(candidates), treat_row, fallback, candidates))

    return pending, n_candidates_by_treat, in_caliper_control_ids


def _print_waterfall_pools(pending, n_treat):
    counts = Counter(fallback for _, _, fallback, _ in pending)
    parts = [
        f"{WATERFALL_POOL_LABELS[label]}={counts.get(label, 0)}"
        for label, _ in WATERFALL_POOLS
    ]
    n_unmatched = n_treat - len(pending)
    print("  Neighborhood pools: " + ", ".join(parts) + f", unmatched={n_unmatched}")


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


def _assign_greedy(pending, k, cap):
    """Hardest-to-match treatment cells first; each takes its nearest available controls.

    Returns [(treat_row, fallback, [(dist, control_id), ...]), ...].
    """
    control_uses = Counter()
    assigned = []
    for _, treat_row, fallback, candidates in sorted(pending, key=lambda item: item[0]):
        chosen = []
        for dist, control_id in candidates:
            if control_uses[control_id] >= cap:
                continue
            control_uses[control_id] += 1
            chosen.append((dist, control_id))
            if len(chosen) >= k:
                break
        assigned.append((treat_row, fallback, chosen))
    return assigned


def _assign_optimal(
    pending,
    k,
    cap,
    max_candidates=OPTIMAL_MAX_CANDIDATES,
    std_covariates=None,
    balance_smd_target=None,
):
    """Assign all candidates at once, under the same reuse cap, in priority order:
    1. most treatment cells matched
    2. if balance_smd_target is set: every covariate's |SMD| at or below the target, or
       as close as possible (least total excess over the target, summed over covariates)
    3. most matched pairs (up to k per cell), then least total Mahalanobis distance

    SMDs are measured exactly as in the report card: pairs weighted 1/k, differences in
    pre-match pooled SDs. std_covariates holds each cell's covariates divided by that
    SD, indexed by cell_ID. The weighting is exact for k <= 2.

    Solved as integer programs (HiGHS via scipy), each holding the previous optimum
    fixed. If the best match without priority 2 already meets the balance target, it is
    also the best balanced match, so the (slower) balance step is skipped. Only each
    cell's nearest max_candidates candidates are considered, to keep large sites fast.
    Returns the same structure as _assign_greedy.
    """
    balance = balance_smd_target is not None
    if balance and k > 2:
        raise ValueError("Balance-constrained matching supports n_neighbors <= 2")

    edges = [
        (i, dist, control_id)
        for i, (_, _, _, candidates) in enumerate(pending)
        for dist, control_id in candidates[:max_candidates]  # sorted by distance
    ]
    if not edges:
        return [(treat_row, fallback, []) for _, treat_row, fallback, _ in pending]

    n_treat, n_edges = len(pending), len(edges)
    treat_idx = np.array([e[0] for e in edges])
    dists = np.array([e[1] for e in edges], dtype=float)
    control_ids, control_idx = np.unique([e[2] for e in edges], return_inverse=True)
    edge_range = np.arange(n_edges)
    ones = np.ones(n_edges)

    # Variables, in order:
    #   x: one per candidate pair, 1 = matched
    #   y: one per treatment cell, 1 = has at least one match
    # and with balance constraints:
    #   u: one per candidate pair, = x * z (linearized)
    #   z: one per treatment cell, 1 = has two matches
    #   s: one per covariate, |SMD| excess over the target
    n_balance_covs = std_covariates.shape[1] if balance else 0
    sizes = {"x": n_edges, "y": n_treat}
    if balance:
        sizes.update(u=n_edges, z=n_treat, s=n_balance_covs)
    starts = dict(zip(sizes, np.cumsum([0, *sizes.values()])))
    n_vars = sum(sizes.values())
    var = {name: slice(starts[name], starts[name] + n) for name, n in sizes.items()}

    def rows(n_rows, **blocks):
        """Constraint rows over all variables, from {variable name: block matrix}."""
        empty = {name: csr_matrix((n_rows, n)) for name, n in sizes.items()}
        return hstack([blocks.get(name, empty[name]) for name in sizes]).tocsr()

    def on(name, values=1.0):
        """Cost or constraint vector with values on one variable and 0 elsewhere."""
        vec = np.zeros(n_vars)
        vec[var[name]] = values
        return vec

    treat_x = csr_matrix((ones, (treat_idx, edge_range)), shape=(n_treat, n_edges))
    control_x = csr_matrix(
        (ones, (control_idx, edge_range)), shape=(len(control_ids), n_edges)
    )
    constraints = [
        # At most k per cell
        LinearConstraint(rows(n_treat, x=treat_x), ub=k),
        # y <= matches
        LinearConstraint(rows(n_treat, x=-treat_x, y=identity(n_treat)), ub=0),
        # Reuse cap
        LinearConstraint(rows(len(control_ids), x=control_x), ub=cap),
    ]
    integrality = on("x") + on("y")
    upper = np.ones(n_vars)

    if balance:
        cell_of_pair = treat_x.T.tocsr()
        pair_eye, cell_eye = identity(n_edges), identity(n_treat)
        constraints += [
            # x <= y
            LinearConstraint(rows(n_edges, x=pair_eye, y=-cell_of_pair), ub=0),
            # z = matches - y, i.e. 1 if the cell has two matches
            LinearConstraint(
                rows(n_treat, x=treat_x, y=-cell_eye, z=-cell_eye), lb=0, ub=0
            ),
            # u <= x, u <= z, u >= x + z - 1
            LinearConstraint(rows(n_edges, u=pair_eye, x=-pair_eye), ub=0),
            LinearConstraint(rows(n_edges, u=pair_eye, z=-cell_of_pair), ub=0),
            LinearConstraint(
                rows(n_edges, u=-pair_eye, x=pair_eye, z=cell_of_pair), ub=1
            ),
        ]
        upper[var["s"]] = np.inf
        # Treatment minus control in pooled SDs: one row per pair, one column per
        # covariate
        treat_ids = [pending[i][1].cell_ID for i in treat_idx]
        diffs = (
            std_covariates.loc[treat_ids].to_numpy()
            - std_covariates.loc[control_ids[control_idx]].to_numpy()
        )
        # Keep float round-off on the passing side of the report card threshold
        target = balance_smd_target - 1e-6

    def smd_excess(x):
        """Each covariate's |SMD| excess over the target for a 0/1 pair selection x,
        with pairs weighted 1/(matches of their treatment cell)."""
        matches = (treat_x @ x)[treat_idx]
        weights = np.divide(x, matches, out=np.zeros(n_edges), where=x > 0)
        smd = weights @ diffs / n_covered
        return np.maximum(np.abs(smd) - target, 0)

    def solve(cost, previous=None):
        """Best solution within the node and time limits; if none was found, the
        previous step's solution, which still satisfies every earlier priority."""
        result = milp(
            cost,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(0, upper),
            options={
                "node_limit": OPTIMAL_NODE_LIMIT,
                "time_limit": OPTIMAL_TIME_LIMIT_S,
            },
        )
        if result.x is not None:
            return result.x
        if previous is None:
            raise RuntimeError(f"Optimal matching failed: {result.message}")
        return previous

    # 1. Maximize matched treatment cells, then hold that fixed
    solution = solve(-on("y"))
    n_covered = round(solution[var["y"]].sum())
    constraints.append(LinearConstraint(on("y"), lb=n_covered))

    # 3. (objective) Maximize matched pairs, then minimize total distance, in one
    #    solve: each pair is worth 1 and all distances together are worth less than 1
    eps = 1 / (k * n_treat * dists.max() + 1)
    pairs_cost = on("x", -1 + eps * dists)

    # Shortcut: if the best match without the balance step already meets the target,
    # no balanced match can beat it on pairs or distance, so it is the answer
    skip_balance = False
    if balance:
        solution = solve(pairs_cost, previous=solution)
        x = np.round(solution[var["x"]])
        skip_balance = not smd_excess(x).any()

    # 2. Minimize total |SMD| excess over the target, then hold that fixed.
    #    Pair weight w = x - u/2 is 1 if its cell has one match and 1/2 if two, so
    #    SMD_j = sum(w * diffs_j) / n_covered, and |SMD_j| <= target + s_j
    if balance and not skip_balance:
        d = csr_matrix(diffs.T)
        excess = -n_covered * identity(n_balance_covs)
        for sign in (1, -1):
            smd_rows = rows(n_balance_covs, x=sign * d, u=-sign * d / 2, s=excess)
            constraints.append(LinearConstraint(smd_rows, ub=target * n_covered))
        solution = solve(on("s"), previous=solution)
        total_excess = smd_excess(np.round(solution[var["x"]])).sum()
        constraints.append(LinearConstraint(on("s"), ub=total_excess + 1e-7))

    # 3. Maximize matched pairs, then minimize total distance
    if not skip_balance:
        solution = solve(pairs_cost, previous=solution)
        x = np.round(solution[var["x"]])

    chosen = [[] for _ in range(n_treat)]
    for e in np.flatnonzero(x > 0.5):
        chosen[treat_idx[e]].append((dists[e], control_ids[control_idx[e]]))
    return [
        (treat_row, fallback, sorted(chosen[i]))
        for i, (_, treat_row, fallback, _) in enumerate(pending)
    ]


def match_treatment_control_mdm(
    cells_df,
    covariates=None,
    caliper=CALIPER_MDM,
    n_neighbors=N_NEIGHBORS_MDM,
    reuse_frac=MAX_CONTROL_REUSE_FRAC,
    reuse_ceiling=MAX_CONTROL_REUSE_CEILING,
    per_covariate_calipers=None,
    matching_algorithm=MATCHING_ALGORITHM,
    balance_smd_target=BALANCE_SMD_TARGET,
):
    """Match each treatment cell to control cells by Mahalanobis Distance Matching (MDM).

    per_covariate_calipers: {covariate: n_sd}; controls must also be within
    n_sd pooled SDs of the treatment cell on each listed covariate.
    matching_algorithm: "greedy" or "optimal" (see _assign_greedy / _assign_optimal).
    balance_smd_target: optimal matching aims for every covariate's |SMD| at or below
    this, after maximizing matched treatment cells. None (or greedy matching) turns
    balance constraints off.
    """
    if covariates is None:
        covariates = COVARIATES
    if per_covariate_calipers is None:
        per_covariate_calipers = PER_COVARIATE_CALIPERS

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

    # Convert SD multiples to raw units, using the same pooled SD as the SMD diagnostic.
    # Pooled SD is NaN with a single treatment cell, so fall back to the control SD.
    max_abs_diff = {}
    for col, n_sd in per_covariate_calipers.items():
        sd = calc_pooled_sd(treat_df[col], control_df[col])
        if not np.isfinite(sd):
            sd = control_df[col].std()
        max_abs_diff[col] = n_sd * sd
    print(f"Per-covariate calipers (SD): {per_covariate_calipers}")

    scaled_cols = [f"_scaled_{c}" for c in covariates]
    pending, n_candidates_by_treat, in_caliper_control_ids = (
        waterfall_in_caliper_neighbors(
            treat_df,
            control_df,
            scaled_cols,
            caliper,
            {"metric": "mahalanobis", "metric_params": {"VI": inv_cov}},
            max_abs_diff,
        )
    )
    _print_waterfall_pools(pending, len(treat_df))

    if matching_algorithm == "greedy":
        assigned = _assign_greedy(pending, n_neighbors, cap)
        balance_smd_target = None
    elif matching_algorithm == "optimal":
        # Balance is measured as in the report card SMD: covariates in pre-match pooled
        # SDs. Covariates with no variation can't be imbalanced, so they are skipped
        std_covariates = None
        if balance_smd_target is not None:
            all_cells = pd.concat([treat_df, control_df]).set_index("cell_ID")
            pooled_sds = pd.Series(
                [calc_pooled_sd(treat_df[col], control_df[col]) for col in covariates],
                index=covariates,
            )
            balance_cols = pooled_sds.index[np.isfinite(pooled_sds) & (pooled_sds > 0)]
            if len(balance_cols):
                std_covariates = all_cells[balance_cols] / pooled_sds[balance_cols]
            else:
                balance_smd_target = None
        assigned = _assign_optimal(
            pending,
            n_neighbors,
            cap,
            std_covariates=std_covariates,
            balance_smd_target=balance_smd_target,
        )
    else:
        raise ValueError(f"Unknown matching_algorithm: {matching_algorithm!r}")
    print(
        f"Matching algorithm: {matching_algorithm}; balance |SMD| target: {balance_smd_target}"
    )

    matches = []
    for treat_row, fallback, chosen in assigned:
        for rank, (dist, control_id) in enumerate(chosen, start=1):
            control_uses[control_id] = control_uses.get(control_id, 0) + 1
            matches.append(
                {
                    "treat_cell_id": treat_row.cell_ID,
                    "control_cell_id": control_id,
                    "mahalanobis_distance": float(dist),
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
                "mahalanobis_distance",
                "match_rank",
                "match_country",
                "match_ecoregion",
                "match_fallback",
            ]
        )

    n_in_caliper = len(in_caliper_control_ids)
    n_treat = len(treat_df)
    demand = n_treat * n_neighbors
    control_supply_ratio = (n_in_caliper * cap) / demand if demand else np.nan
    match_df.attrs["n_in_caliper_controls"] = n_in_caliper
    match_df.attrs["reuse_cap"] = cap
    match_df.attrs["n_neighbors"] = n_neighbors
    match_df.attrs["n_candidates_by_treat"] = n_candidates_by_treat

    n_treat_counts = len(n_candidates_by_treat)
    counts = np.fromiter(
        n_candidates_by_treat.values(), dtype=float, count=n_treat_counts
    )
    frac0 = float((counts == 0).mean()) if n_treat_counts else np.nan
    frac1 = float((counts == 1).mean()) if n_treat_counts else np.nan
    frac2 = float((counts >= 2).mean()) if n_treat_counts else np.nan

    print("\nResults:")
    print(f"  Treatment cells matched: {match_df['treat_cell_id'].nunique()}")
    print(f"  Unique control cells used: {match_df['control_cell_id'].nunique()}")
    print(f"  Total matched pairs: {len(match_df)}")
    print(f"  In-caliper neighbors: 0={frac0:.1%}, 1={frac1:.1%}, 2+={frac2:.1%}")
    print(
        f"  In-caliper controls: {n_in_caliper}; "
        f"control supply ratio: {control_supply_ratio:.2f} "
        f"(in-caliper × cap) / (n_treat × k)"
    )
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
