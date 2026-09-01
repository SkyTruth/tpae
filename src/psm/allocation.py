"""
Sample budget allocation for global propensity sampling.

Two-stage allocation:
  1. Global: how many samples per stratum (protected × biome) globally?
     Computed once from biome areas + treatment/control ratio.
  2. Per-tile: given a tile and the global per-stratum budget, how many
     samples of each stratum should come from this tile? Proportional to
     the stratum's pixel count within the tile.

Strata convention (matches existing notebook):
    stratum_id = protected * 20 + biome_num
    where protected in {0, 1} and biome_num in {1, ..., 14}.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd

from utils.variables import EE_CRS_METERS

BIOME_FC_ASSET = "RESOLVE/ECOREGIONS/2017"
N_BIOMES = 14  # RESOLVE biomes 1-14 (terrestrial)


def compute_global_allocation(
    total_points: int = 100000,
    treat_control_ratio: tuple[int, int] = (1, 2),
    min_per_stratum: int = 500,
) -> dict[int, int]:
    """
    Compute global per-stratum sample budget using equal-per-biome allocation.

    Each terrestrial biome receives an equal share of the total budget,
    split between protected and unprotected pools according to
    treat_control_ratio. This prioritizes uniform model performance
    across biomes over global area-proportional representation.

    Rationale: the propensity model will be used to compute relative
    effectiveness for individual PAs and for cross-biome aggregations.
    Equal-per-biome allocation produces uniform matching quality across
    biomes, preventing aggregation bias toward biomes with better-fit
    models. Area-proportional allocation (as in Li et al. 2024) would
    optimize for global average effects but degrade small-biome performance.

    Parameters
    ----------
    total_points : int
        Total samples globally across all strata.
    treat_control_ratio : tuple[int, int]
        Treatment:control ratio. (1, 2) = 1/3 treatment, 2/3 control.
    min_per_stratum : int
        Floor on samples per (protected, biome) stratum. With equal-per-
        biome allocation this floor rarely activates, but kept for safety.

    Returns
    -------
    dict[int, int]
        Mapping stratum_id -> sample count. Sum is approximately total_points.
    """
    treat_frac = treat_control_ratio[0] / sum(treat_control_ratio)
    treatment_budget = int(total_points * treat_frac)
    control_budget = total_points - treatment_budget

    # Equal share per biome within each protection class
    treatment_per_biome = treatment_budget // N_BIOMES
    control_per_biome = control_budget // N_BIOMES

    allocation: dict[int, int] = {}
    for biome_num in range(1, N_BIOMES + 1):
        allocation[int(0 * 20 + biome_num)] = max(control_per_biome, min_per_stratum)
        allocation[int(1 * 20 + biome_num)] = max(treatment_per_biome, min_per_stratum)

    return allocation


def _frequency_histogram_for_tile(
    tile_id: str,
    tile_geom: ee.Geometry,
    strata_image: ee.Image,
    scale: int,
    projection: ee.Projection,
) -> tuple[str, dict[int, int]]:
    """
    Per-tile stratum pixel count. Used by compute_tile_allocations.
    """
    hist = strata_image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=tile_geom,
        scale=scale,
        crs=projection,
        maxPixels=1e10,
        bestEffort=False,
    ).getInfo()

    raw = hist.get("strata") or {}
    counts = {int(k): int(v) for k, v in raw.items()}
    return tile_id, counts


def compute_tile_pixel_counts(
    tiles: dict[str, ee.Geometry],
    strata_image: ee.Image,
    scale: int,
    projection: ee.Projection,
    max_workers: int = 10,
) -> dict[str, dict[int, int]]:
    """
    Count pixels per stratum within each tile.

    Parallelized with a thread pool — each task is an independent EE
    request, so threading helps despite the GIL.

    Parameters
    ----------
    tiles : dict[str, ee.Geometry]
        Output of filter_tiles_to_land().
    strata_image : ee.Image
        Single-band image where each pixel value is stratum_id.
    scale : int
        Sampling scale (typically PSM_CELL_SIZE = 1000 m).
    projection : ee.Projection
        EPSG:6933 at the target scale.
    max_workers : int
        Concurrent EE requests. EE's interactive quota typically allows
        ~20-40 concurrent getInfo calls; 10 is conservative.

    Returns
    -------
    dict[tile_id, dict[stratum_id, pixel_count]]
    """
    tile_counts: dict[str, dict[int, int]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _frequency_histogram_for_tile,
                tile_id,
                geom,
                strata_image,
                scale,
                projection,
            ): tile_id
            for tile_id, geom in tiles.items()
        }
        for future in as_completed(futures):
            tile_id, counts = future.result()
            tile_counts[tile_id] = counts

    return tile_counts


def compute_tile_allocations(
    tile_pixel_counts: dict[str, dict[int, int]],
    global_allocation: dict[int, int],
    min_tile_stratum: int = 1,
) -> dict[str, dict[int, int]]:
    """
    Distribute the global per-stratum budget across tiles.

    For each stratum, each tile receives a share proportional to the
    stratum's pixel count in that tile. Sum across tiles equals the
    global budget for that stratum (within rounding).

    Parameters
    ----------
    tile_pixel_counts : dict
        Output of compute_tile_pixel_counts().
    global_allocation : dict
        Output of compute_global_allocation().
    min_tile_stratum : int
        Tiles with fewer than this many requested samples for a stratum
        will have that stratum dropped from their request (avoids tiny
        per-stratum requests that waste task overhead).

    Returns
    -------
    dict[tile_id, dict[stratum_id, sample_count]]
        Empty inner dicts mean the tile has no samples to draw.
    """
    # Sum pixel counts across tiles to get global denominator per stratum
    global_pixel_count: dict[int, int] = {}
    for counts in tile_pixel_counts.values():
        for stratum_id, count in counts.items():
            global_pixel_count[stratum_id] = global_pixel_count.get(stratum_id, 0) + count

    tile_alloc: dict[str, dict[int, int]] = {}
    for tile_id, counts in tile_pixel_counts.items():
        per_stratum: dict[int, int] = {}
        for stratum_id, tile_count in counts.items():
            if stratum_id not in global_allocation:
                # Stratum exists in pixels but not in budget (e.g., biome 0
                # leaked into the raster). Skip.
                continue
            global_count = global_pixel_count.get(stratum_id, 0)
            if global_count == 0:
                continue
            share = tile_count / global_count
            n = int(round(global_allocation[stratum_id] * share))
            if n >= min_tile_stratum:
                per_stratum[stratum_id] = n
        tile_alloc[tile_id] = per_stratum

    return tile_alloc


def validate_allocation(
    tile_allocations: dict[str, dict[int, int]],
    global_allocation: dict[int, int],
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """
    Sanity check: per-stratum sum across tiles should be close to global budget.

    Returns a DataFrame of (stratum_id, global_target, tile_sum, ratio).
    Rounding can cause small mismatches; large deviations indicate a bug.
    """
    tile_sums: dict[int, int] = {}
    for per_stratum in tile_allocations.values():
        for stratum_id, n in per_stratum.items():
            tile_sums[stratum_id] = tile_sums.get(stratum_id, 0) + n

    rows = []
    for stratum_id, target in global_allocation.items():
        actual = tile_sums.get(stratum_id, 0)
        rows.append(
            {
                "stratum_id": stratum_id,
                "protected": stratum_id // 20,
                "biome_num": stratum_id % 20,
                "global_target": target,
                "tile_sum": actual,
                "ratio": actual / target if target > 0 else float("nan"),
            }
        )
    df = pd.DataFrame(rows).sort_values("stratum_id")

    bad = df[(df["ratio"] - 1).abs() > tolerance]
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} strata deviate from global budget by >{tolerance:.1%}")
        print(bad)
    return df


def save_allocation(allocation: dict, path: str | Path) -> None:
    """Persist allocation dict to JSON for reproducibility / caching."""
    # JSON keys must be strings
    serializable = {str(k): v for k, v in allocation.items()}
    Path(path).write_text(json.dumps(serializable, indent=2))


def load_allocation(path: str | Path) -> dict:
    """Load allocation dict from JSON, casting keys back to int."""
    raw = json.loads(Path(path).read_text())
    return {int(k): v for k, v in raw.items()}
