"""
Propensity score prediction using the saved propensity model.

This model is a logistic regression with biome × covariate interactions.
Its design matrix has 83 features built in a specific order:
    [5 standardized covariates, 13 biome dummies, 65 interaction terms]

This module reconstructs that exact feature ordering for new data and computes
propensity scores. Getting the column order wrong silently produces meaningless
predictions, so this is the only safe way to call the saved model.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_propensity_artifacts(path: str | Path) -> dict[str, Any]:
    """
    Load the saved model artifacts dict.

    Returns
    -------
    dict with keys: model, scaler, covariates, biome_levels_ref,
    biome_levels_dummied, feature_names, training_metadata
    """
    return joblib.load(path)


def build_design_matrix(
    df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> np.ndarray:
    """
    Reconstruct the design matrix for new pixels.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: COVARIATES (e.g., elevation, slope, ...) + 'biome'.
        Biome values must be integers in 1..14.
    artifacts : dict
        Loaded propensity model artifacts.

    Returns
    -------
    np.ndarray of shape (n_rows, n_features) in the same column order as training.

    Raises
    ------
    ValueError if required columns are missing or biome values are out of range.
    """
    covariates = artifacts["covariates"]
    scaler = artifacts["scaler"]
    biome_ref = artifacts["biome_levels_ref"]
    biome_dummied = artifacts["biome_levels_dummied"]
    expected_feature_names = artifacts["feature_names"]

    # Validate inputs
    missing_cols = [c for c in covariates + ["biome"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[covariates + ["biome"]].isna().any().any():
        na_counts = df[covariates + ["biome"]].isna().sum()
        raise ValueError(f"NaN values in input:\n{na_counts[na_counts > 0]}")

    biome_values = df["biome"].astype(int).values
    valid_biomes = set([biome_ref] + list(biome_dummied))
    invalid = set(biome_values) - valid_biomes
    if invalid:
        raise ValueError(
            f"Biome values not seen in training: {invalid}. "
            f"Valid biomes: {sorted(valid_biomes)}"
        )

    # Standardize covariates using the training scaler
    X_scaled = scaler.transform(df[covariates].values)

    # One-hot encode biome with the same reference and ordering as training
    n_rows = len(df)
    n_dummies = len(biome_dummied)
    biome_dummies = np.zeros((n_rows, n_dummies), dtype=float)
    for i, b in enumerate(biome_dummied):
        biome_dummies[:, i] = (biome_values == b).astype(float)

    # Build interaction terms: covariate × biome dummy
    # Order must match training: for each biome (in biome_dummied order),
    # for each covariate (in covariates order), make a column
    n_main = len(covariates)
    interactions = np.zeros((n_rows, n_dummies * n_main), dtype=float)
    for i in range(n_dummies):
        for j in range(n_main):
            interactions[:, i * n_main + j] = X_scaled[:, j] * biome_dummies[:, i]

    X_full = np.hstack([X_scaled, biome_dummies, interactions])

    # Sanity check: feature count must match training
    if X_full.shape[1] != len(expected_feature_names):
        raise RuntimeError(
            f"Feature count mismatch: built {X_full.shape[1]}, "
            f"expected {len(expected_feature_names)}. "
            f"This is a bug in build_design_matrix."
        )

    return X_full


def predict_propensity(
    df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> np.ndarray:
    """
    Compute propensity scores P(protected=1 | covariates) for new pixels.

    Parameters
    ----------
    df : DataFrame
        Must contain COVARIATES + 'biome' columns. Other columns are ignored.
    artifacts : dict
        Loaded from load_propensity_artifacts().

    Returns
    -------
    np.ndarray of length n_rows, with propensity scores in [0, 1].
    """
    X = build_design_matrix(df, artifacts)
    return artifacts["model"].predict_proba(X)[:, 1]
    