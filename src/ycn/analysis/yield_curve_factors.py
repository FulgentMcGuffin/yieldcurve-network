"""Yield curve factor models: Nelson-Siegel, PCA, and multi-layer decomposition."""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import polars as pl
from scipy.linalg import svd
from scipy.optimize import least_squares, minimize
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.mixture import GaussianMixture


class NSFactors(NamedTuple):
    """Nelson-Siegel factor estimates."""

    level: float  # β₀
    slope: float  # β₁
    curvature: float  # β₂
    decay: float  # λ
    rmse: float  # fit residual error
    residuals: np.ndarray  # yield - fitted


class PCAFactors(NamedTuple):
    """PCA factor decomposition."""

    scores: np.ndarray  # (n_dates, n_components)
    loadings: np.ndarray  # (n_terms, n_components)
    variance_explained: np.ndarray  # (n_components,)
    residuals: np.ndarray  # reconstructed residuals


class YieldCurveRegime(NamedTuple):
    """Regime classification result."""

    regime_labels: np.ndarray  # (n_dates,) cluster labels
    regime_means: np.ndarray  # (n_regimes, n_features) cluster centers
    regime_probs: np.ndarray  # (n_dates, n_regimes) posterior probabilities
    regime_names: list[str]  # human-readable names


# Nelson-Siegel basis functions


def ns_basis(
    maturities: np.ndarray, decay: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Nelson-Siegel basis functions for given maturities.

    Args:
        maturities: Array of time-to-maturity values (years).
        decay: Decay parameter λ controlling speed of convergence to long-term rate.

    Returns:
        (f1, f2, f3) basis functions as arrays of shape (n_maturities,).
    """
    m = np.asarray(maturities, dtype=float)
    lambda_m = decay

    # Avoid division by zero
    m_safe = np.where(m < 1e-6, 1e-6, m)

    exp_term = np.exp(-m_safe / lambda_m)

    # Level basis: always 1
    f0 = np.ones_like(m)

    # Slope basis: (1 - exp(-m/λ)) / (m/λ)
    f1 = (1.0 - exp_term) / (m_safe / lambda_m)

    # Curvature basis: [(1 - exp(-m/λ)) / (m/λ) - exp(-m/λ)]
    f2 = (1.0 - exp_term) / (m_safe / lambda_m) - exp_term

    return f0, f1, f2


def fit_nelson_siegel(
    maturities: np.ndarray,
    yields: np.ndarray,
    decay: float | None = None,
    decay_bounds: tuple[float, float] = (0.05, 2.0),
) -> NSFactors:
    """Fit Nelson-Siegel model to yield curve.

    Args:
        maturities: Array of maturities (years).
        yields: Array of yields at corresponding maturities.
        decay: Fixed decay parameter. If None, optimizes via grid search + refinement.
        decay_bounds: Search bounds for decay parameter optimization.

    Returns:
        NSFactors with estimated level, slope, curvature.
    """
    yields = np.asarray(yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    # Grid search for optimal decay if not provided
    if decay is None:
        decays = np.linspace(decay_bounds[0], decay_bounds[1], 50)
        best_rmse = float("inf")
        best_decay = decays[0]

        for d in decays:
            f0, f1, f2 = ns_basis(maturities, d)
            X = np.column_stack([f0, f1, f2])
            try:
                beta = np.linalg.lstsq(X, yields, rcond=None)[0]
                fitted = X @ beta
                rmse = np.sqrt(np.mean((yields - fitted) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_decay = d
            except np.linalg.LinAlgError:
                continue

        decay = best_decay

    # Fit with fixed decay
    f0, f1, f2 = ns_basis(maturities, decay)
    X = np.column_stack([f0, f1, f2])
    beta = np.linalg.lstsq(X, yields, rcond=None)[0]

    fitted = X @ beta
    residuals = yields - fitted
    rmse = np.sqrt(np.mean(residuals**2))

    return NSFactors(
        level=float(beta[0]),
        slope=float(beta[1]),
        curvature=float(beta[2]),
        decay=float(decay),
        rmse=float(rmse),
        residuals=residuals,
    )


def fit_nelson_siegel_batch(
    df_wide: pl.DataFrame,
    date_col: str,
    maturities: list[str],
    decay: float | None = None,
) -> pl.DataFrame:
    """Fit Nelson-Siegel model across multiple dates.

    Args:
        df_wide: DataFrame with rows=dates, cols=maturities.
        date_col: Name of date column.
        maturities: Maturity labels (e.g., ['0.5Y', '1Y', '2Y', ...]).
        decay: Fixed decay parameter or None to optimize per-date.

    Returns:
        DataFrame with columns: date, level, slope, curvature, rmse.
    """
    # Convert maturity labels to numeric (assumes format like "0.5Y", "1Y", etc.)
    maturity_values = []
    for m in maturities:
        m_str = m.replace("Y", "").replace("p", ".")
        try:
            maturity_values.append(float(m_str))
        except ValueError:
            raise ValueError(f"Cannot parse maturity label: {m}")

    maturity_array = np.array(maturity_values)

    results = []
    for row in df_wide.iter_rows(named=True):
        date_val = row[date_col]
        yields_val = np.array([row[m] for m in maturities], dtype=float)

        # Skip if any NaN
        if np.any(np.isnan(yields_val)):
            continue

        try:
            ns = fit_nelson_siegel(maturity_array, yields_val, decay=decay)
            results.append(
                {
                    date_col: date_val,
                    "level": ns.level,
                    "slope": ns.slope,
                    "curvature": ns.curvature,
                    "decay": ns.decay,
                    "rmse": ns.rmse,
                }
            )
        except Exception:
            continue

    return pl.DataFrame(results)


# PCA Approach


def fit_pca(
    yields_matrix: np.ndarray,
    n_components: int = 3,
) -> PCAFactors:
    """Fit PCA to yield matrix (dates × maturities).

    Args:
        yields_matrix: (n_dates, n_maturities) array of yields.
        n_components: Number of principal components to extract.

    Returns:
        PCAFactors with scores, loadings, and explained variance.
    """
    # Standardize
    yields_std = (yields_matrix - yields_matrix.mean(axis=0)) / yields_matrix.std(
        axis=0
    )

    # Fit PCA
    pca = SklearnPCA(n_components=n_components)
    scores = pca.fit_transform(yields_std)
    loadings = pca.components_.T

    # Reconstruct
    yields_hat = scores @ loadings.T
    residuals = yields_std - yields_hat

    return PCAFactors(
        scores=scores,
        loadings=loadings,
        variance_explained=pca.explained_variance_ratio_,
        residuals=residuals,
    )


# Regime Classification


def classify_yield_curve_regimes(
    factors: np.ndarray,  # (n_dates, n_factors): level, slope, curvature
    n_regimes: int = 3,
) -> YieldCurveRegime:
    """Classify yield curve into regimes using Gaussian Mixture Model.

    Args:
        factors: (n_dates, n_factors) array of yield curve factors.
        n_regimes: Number of regimes to identify.

    Returns:
        YieldCurveRegime with labels, centers, and probabilities.
    """
    gmm = GaussianMixture(n_components=n_regimes, random_state=0, n_init=10)
    labels = gmm.fit_predict(factors)
    probs = gmm.predict_proba(factors)

    # Name regimes based on characteristics
    regime_names = []
    for k in range(n_regimes):
        mean_factors = gmm.means_[k]
        # Assume factors are: [level, slope, curvature]
        if len(mean_factors) >= 2:
            slope = mean_factors[1]
            if slope > 0.5:
                name = "Steep"
            elif slope < -0.5:
                name = "Inverted"
            else:
                name = "Flat"
        else:
            name = f"Regime {k}"
        regime_names.append(name)

    return YieldCurveRegime(
        regime_labels=labels,
        regime_means=gmm.means_,
        regime_probs=probs,
        regime_names=regime_names,
    )


# Multi-Layer Inter-Layer Weighting


def compute_interlayer_weights(
    ns_factors_by_issuer: dict[
        str, list[NSFactors]
    ],  # issuer -> [factors_t1, factors_t2, ...]
    term_distance_weight: float = 0.3,
    factor_loading_weight: float = 0.7,
) -> dict[tuple[str, str], float]:
    """Compute inter-layer weights based on NS factor loadings and term structure.

    Args:
        ns_factors_by_issuer: Dict mapping issuer to list of NS factor objects.
        term_distance_weight: Weight for term-distance component in edge weight.
        factor_loading_weight: Weight for factor-loading component.

    Returns:
        Dict mapping (issuer, term1_idx, term2_idx) -> edge weight.
    """
    weights = {}

    for issuer, factors_list in ns_factors_by_issuer.items():
        if len(factors_list) == 0:
            continue

        # Average factors across time
        avg_level = np.mean([f.level for f in factors_list])
        avg_slope = np.mean([f.slope for f in factors_list])
        avg_curvature = np.mean([f.curvature for f in factors_list])

        # Create inter-layer edge weight combining:
        # 1. Factor loading magnitude (how much does issuer's curve bend?)
        factor_effect = np.abs(avg_slope) + np.abs(avg_curvature)

        # 2. Normalize
        weight = (
            factor_loading_weight * factor_effect / (1 + factor_effect)
            + term_distance_weight * 0.5
        )

        weights[issuer] = float(np.clip(weight, 0, 1))

    return weights


# Residual Network Construction


def extract_ns_residuals(
    df_wide: pl.DataFrame,
    date_col: str,
    term_cols: list[str],
    decay: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Extract Nelson-Siegel residuals from yield curves.

    Args:
        df_wide: DataFrame with yields (rows=dates, cols=terms).
        date_col: Name of date column.
        term_cols: Column names for maturities.
        decay: Fixed decay parameter or None to optimize.

    Returns:
        (residuals_matrix, metadata) where residuals_matrix is (n_dates, n_terms).
    """
    maturity_values = []
    for m in term_cols:
        m_str = m.replace("Y", "").replace("p", ".")
        try:
            maturity_values.append(float(m_str))
        except ValueError:
            raise ValueError(f"Cannot parse maturity: {m}")

    maturity_array = np.array(maturity_values)
    residuals_list = []
    dates_valid = []

    for row in df_wide.iter_rows(named=True):
        yields_val = np.array([row[m] for m in term_cols], dtype=float)

        if np.any(np.isnan(yields_val)):
            continue

        try:
            ns = fit_nelson_siegel(maturity_array, yields_val, decay=decay)
            residuals_list.append(ns.residuals)
            dates_valid.append(row[date_col])
        except Exception:
            continue

    residuals_matrix = np.array(residuals_list)

    return residuals_matrix, {
        "dates": dates_valid,
        "maturities": term_cols,
        "maturity_values": maturity_array,
    }


# Temporal Dynamics Tracking


def compute_factor_trajectories(
    factors_df: pl.DataFrame,
    window_size: int = 20,
    step_size: int = 5,
) -> dict:
    """Compute rolling-window statistics on factor trajectories.

    Args:
        factors_df: DataFrame with columns: date, level, slope, curvature.
        window_size: Rolling window size (days).
        step_size: Step size between windows.

    Returns:
        Dict with statistics: mean, std, min, max per factor per window.
    """
    factors_np = factors_df.select(["level", "slope", "curvature"]).to_numpy()
    dates = factors_df.select("date").to_numpy().flatten()

    stats = {
        "windows": [],
        "level_mean": [],
        "level_std": [],
        "slope_mean": [],
        "slope_std": [],
        "curvature_mean": [],
        "curvature_std": [],
    }

    for i in range(0, len(factors_np) - window_size + 1, step_size):
        window_factors = factors_np[i : i + window_size]

        stats["windows"].append((dates[i], dates[i + window_size - 1]))
        stats["level_mean"].append(float(np.mean(window_factors[:, 0])))
        stats["level_std"].append(float(np.std(window_factors[:, 0])))
        stats["slope_mean"].append(float(np.mean(window_factors[:, 1])))
        stats["slope_std"].append(float(np.std(window_factors[:, 1])))
        stats["curvature_mean"].append(float(np.mean(window_factors[:, 2])))
        stats["curvature_std"].append(float(np.std(window_factors[:, 2])))

    return stats


# Signal Subgraph Detection (via node-based importance)


def compute_node_factor_importance(
    centrality_matrix: np.ndarray,  # (n_nodes, n_factors) or (n_nodes, n_times)
) -> dict[str, float]:
    """Compute node importance based on factor sensitivity.

    Args:
        centrality_matrix: (n_nodes, n_features) array of node-feature values.

    Returns:
        Dict mapping node_idx -> importance score.
    """
    # Importance = variance of node's profile across factors/time
    variances = np.var(centrality_matrix, axis=1)
    total_var = np.sum(variances)

    return {i: float(v / total_var) for i, v in enumerate(variances)}
