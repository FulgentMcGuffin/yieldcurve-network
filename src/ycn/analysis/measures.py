"""Pairwise connection measures between node time series."""

from __future__ import annotations

from collections.abc import Callable

import dcor
import numpy as np
import polars as pl
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

ProgressCallback = Callable[[int, int, str], None]

# ace_cream wraps a compiled Fortran extension (gfortran + meson-python), so it
# may fail to import in environments without that toolchain even though it's
# declared as an optional dependency. Probed once at import time (i.e. GUI
# startup) so the "Maximal correlation (ACE)" measure can be hidden rather than
# crashing the app the first time it's selected.
try:
    from ace_cream import ace_cream

    ACE_AVAILABLE = True
    ACE_IMPORT_ERROR: str | None = None
except (
    Exception
) as exc:  # noqa: BLE001 — ImportError, or OSError from a failed DLL/shared-lib load
    ace_cream = None
    ACE_AVAILABLE = False
    ACE_IMPORT_ERROR = str(exc)


def distance_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square distance-correlation matrix for wide-format columns."""
    dcor_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                dcor_val = float("nan")
            else:
                dcor_val = float(dcor.distance_correlation(vi_vj[:, 0], vi_vj[:, 1]))
            dcor_values[i][j] = dcor_val
            dcor_values[j][i] = dcor_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [dcor_values[row][col] for col in nodes] for row in nodes}
    )


def pearson_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square Pearson correlation matrix for wide-format columns.

    Handles NaN values and zero-variance columns gracefully.
    """
    pearson_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                pearson_val = float("nan")
            else:
                # Check for zero variance (correlation undefined)
                if np.std(vi_vj[:, 0]) == 0 or np.std(vi_vj[:, 1]) == 0:
                    pearson_val = float("nan")
                else:
                    corr = np.corrcoef(vi_vj[:, 0], vi_vj[:, 1])[0, 1]
                    # Ensure it's a scalar, not array, and handle NaN
                    pearson_val = float(corr) if not np.isnan(corr) else float("nan")
            pearson_values[i][j] = pearson_val
            pearson_values[j][i] = pearson_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [pearson_values[row][col] for col in nodes] for row in nodes}
    )


def spearman_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square Spearman correlation matrix for wide-format columns.

    Handles NaN values and constant sequences gracefully.
    """
    spearman_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                spearman_val = float("nan")
            else:
                try:
                    corr, _ = stats.spearmanr(vi_vj[:, 0], vi_vj[:, 1])
                    # Ensure it's a scalar and handle NaN
                    spearman_val = float(corr) if not np.isnan(corr) else float("nan")
                except (ValueError, RuntimeError):
                    # Handle edge cases (constant sequences, all NaN, etc.)
                    spearman_val = float("nan")
            spearman_values[i][j] = spearman_val
            spearman_values[j][i] = spearman_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [spearman_values[row][col] for col in nodes] for row in nodes}
    )


def maximal_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Compute a square maximal-correlation (ACE) matrix for wide-format columns.

    Uses the Alternating Conditional Expectations algorithm (Breiman & Friedman,
    1985) to find transformations f, g of x and y that maximize
    corr(f(x), g(y)); the Pearson correlation of the resulting transformed
    variables is the bivariate maximal correlation rho*(x, y) (see
    https://en.wikipedia.org/wiki/Alternating_conditional_expectations,
    "Bivariate case"). Sign is arbitrary under transformation, so the absolute
    value is reported. Handles NaN values, zero-variance columns, and ACE
    convergence failures gracefully.

    Raises:
        RuntimeError: If ace_cream's compiled extension could not be imported
            (e.g. no Fortran compiler was available when it was installed).
            Callers should check ACE_AVAILABLE / available_measures() first.
    """
    if not ACE_AVAILABLE:
        raise RuntimeError(
            "Maximal correlation (ACE) is unavailable: ace_cream failed to "
            f"import ({ACE_IMPORT_ERROR}). It requires a Fortran compiler "
            "(gfortran) at install time -- run `uv sync --extra ace` after "
            "installing one, then restart."
        )
    mcor_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            vi_vj = np.column_stack((v_i, v_j))
            vi_vj = vi_vj[~np.isnan(vi_vj).any(axis=1)]
            if len(vi_vj) < 3:
                mcor_val = float("nan")
            elif np.std(vi_vj[:, 0]) == 0 or np.std(vi_vj[:, 1]) == 0:
                mcor_val = float("nan")
            else:
                try:
                    tx, ty = ace_cream(vi_vj[:, 0], vi_vj[:, 1])
                    corr = np.corrcoef(tx[:, 0], ty[:, 0])[0, 1]
                    mcor_val = float(abs(corr)) if not np.isnan(corr) else float("nan")
                except Exception:
                    # ACE convergence/dimension-overflow failures -> undefined for this pair
                    mcor_val = float("nan")
            mcor_values[i][j] = mcor_val
            mcor_values[j][i] = mcor_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [mcor_values[row][col] for col in nodes] for row in nodes}
    )


def kendall_tau_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Kendall's tau rank correlation matrix (normalized to [0,1] via abs)."""
    tau_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            valid = ~(np.isnan(v_i) | np.isnan(v_j))
            if np.sum(valid) < 3:
                tau_val = float("nan")
            else:
                tau_val, _ = stats.kendalltau(v_i[valid], v_j[valid])
                tau_val = abs(tau_val)  # Normalize to [0, 1]
            tau_values[i][j] = tau_val
            tau_values[j][i] = tau_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame({row: [tau_values[row][col] for col in nodes] for row in nodes})


def dtw_distance_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Dynamic Time Warping distance normalized to [0,1] similarity."""
    from scipy.spatial.distance import euclidean

    def dtw_distance(x, y):
        """Compute DTW distance between two sequences."""
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
                )
        return dtw_matrix[n, m]

    dtw_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    max_dtw = 1e-6
    distances = []

    # First pass: compute all distances to find max
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        v_i = v_i[~np.isnan(v_i)]
        for j in nodes[k:]:
            v_j = df_wide.get_column(j).to_numpy()
            v_j = v_j[~np.isnan(v_j)]
            if len(v_i) < 3 or len(v_j) < 3:
                distances.append(float("nan"))
            else:
                dist = dtw_distance(v_i, v_j)
                max_dtw = max(max_dtw, dist)
                distances.append(dist)
        k += 1

    # Second pass: normalize and store
    dist_idx = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(dist_idx, n_pairs, f"{i} / {j}")
            dist = distances[dist_idx]
            if np.isnan(dist):
                similarity = float("nan")
            else:
                similarity = 1.0 - min(1.0, dist / max_dtw)  # Normalize to [0, 1]
            dtw_values[i][j] = similarity
            dtw_values[j][i] = similarity
            dist_idx += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame({row: [dtw_values[row][col] for col in nodes] for row in nodes})


def fastdtw_distance_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
    radius: int = 2,
) -> pl.DataFrame:
    """FastDTW: Approximate DTW using Sakoe-Chiba band (O(n·radius) vs O(n²)).

    Uses a windowed constraint to reduce computational cost. More accurate
    than raw DTW but much faster. Good for large datasets.

    Args:
        df_wide: Wide-format DataFrame.
        nodes: List of node names.
        progress: Optional progress callback.
        radius: Sakoe-Chiba band radius (window width). Smaller = faster.
                Default 2; try 1-5. radius=1 is very restrictive.

    Returns:
        Normalized distance matrix [0, 1] where larger = more similar.
    """

    def fastdtw_distance(x, y, radius=2):
        """FastDTW with Sakoe-Chiba band constraint."""
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            # Window constraint: from max(1, i - radius) to min(m, i + radius)
            j_min = max(1, i - radius)
            j_max = min(m + 1, i + radius + 1)

            for j in range(j_min, j_max):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )

        return dtw_matrix[n, m]

    dtw_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    max_dtw = 1e-6
    distances = []

    # First pass: compute all distances
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        v_i = v_i[~np.isnan(v_i)]
        for j in nodes[k:]:
            v_j = df_wide.get_column(j).to_numpy()
            v_j = v_j[~np.isnan(v_j)]

            if len(v_i) < 3 or len(v_j) < 3:
                distances.append(float("nan"))
            else:
                dist = fastdtw_distance(v_i, v_j, radius=radius)
                max_dtw = max(max_dtw, dist)
                distances.append(dist)
        k += 1

    # Second pass: normalize and store
    dist_idx = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(dist_idx, n_pairs, f"{i} / {j}")
            dist = distances[dist_idx]
            if np.isnan(dist):
                similarity = float("nan")
            else:
                similarity = 1.0 - min(1.0, dist / max_dtw)  # Normalize to [0, 1]
            dtw_values[i][j] = similarity
            dtw_values[j][i] = similarity
            dist_idx += 1
        k += 1

    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame({row: [dtw_values[row][col] for col in nodes] for row in nodes})


def shrinkage_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Ledoit-Wolf shrinkage correlation matrix (denoised Pearson)."""
    from sklearn.covariance import LedoitWolf

    data = df_wide.select(nodes).to_numpy()
    data = data[~np.isnan(data).any(axis=1)]  # Remove rows with any NaN

    if len(data) < 2:
        return pl.DataFrame({node: [float("nan")] * len(nodes) for node in nodes})

    # Compute Ledoit-Wolf shrunk covariance
    lw = LedoitWolf()
    lw.fit(data)
    cov = lw.covariance_

    # Convert covariance to correlation
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)

    if progress is not None:
        progress(len(nodes), len(nodes), "done")

    # Normalize to [0, 1]
    corr = (corr + 1.0) / 2.0
    return pl.DataFrame({nodes[i]: corr[i].tolist() for i in range(len(nodes))})


def conditional_correlation_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
    quantile: float = 0.90,
) -> pl.DataFrame:
    """Correlation computed only on days where abs(returns) > quantile threshold (stress regime)."""
    corr_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        v_i_clean = v_i[~np.isnan(v_i)]
        threshold_i = np.quantile(np.abs(v_i_clean), quantile)
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            v_j_clean = v_j[~np.isnan(v_j)]
            threshold_j = np.quantile(np.abs(v_j_clean), quantile)

            # Keep only rows where both exceed threshold
            valid = (
                ~np.isnan(v_i)
                & ~np.isnan(v_j)
                & (np.abs(v_i) > threshold_i)
                & (np.abs(v_j) > threshold_j)
            )

            if np.sum(valid) < 3:
                corr_val = float("nan")
            else:
                corr_val = np.corrcoef(v_i[valid], v_j[valid])[0, 1]
                corr_val = (corr_val + 1.0) / 2.0  # Normalize to [0, 1]

            corr_values[i][j] = corr_val
            corr_values[j][i] = corr_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame(
        {row: [corr_values[row][col] for col in nodes] for row in nodes}
    )


def mutual_information_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Mutual information using k-NN estimation, normalized to [0,1]."""
    from sklearn.feature_selection import mutual_info_regression

    mi_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    mi_max = 1e-6  # Running max for normalization
    mi_list = []  # Collect all MI values first

    # First pass: compute all MI values
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            v_j = df_wide.get_column(j).to_numpy()
            valid = ~(np.isnan(v_i) | np.isnan(v_j))
            if np.sum(valid) < 3:
                mi_val = float("nan")
            else:
                mi_val = mutual_info_regression(
                    v_i[valid].reshape(-1, 1), v_j[valid], random_state=0
                )[0]
                mi_max = max(mi_max, float(mi_val))
                mi_list.append(mi_val)
        k += 1

    # Second pass: normalize and store
    mi_idx = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(mi_idx, len(mi_list), f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            valid = ~(np.isnan(v_i) | np.isnan(v_j))
            if np.sum(valid) < 3:
                mi_val = float("nan")
            else:
                mi_val = min(1.0, mi_list[mi_idx] / mi_max)  # Normalize to [0, 1]
            mi_values[i][j] = mi_val
            mi_values[j][i] = mi_val
            mi_idx += 1
        k += 1

    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame({row: [mi_values[row][col] for col in nodes] for row in nodes})


def chatterjee_xi_matrix(
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
) -> pl.DataFrame:
    """Chatterjee's xi correlation (rank-based, detects any dependence)."""
    from scipy.stats import rankdata

    xi_values = {node: {} for node in nodes}
    n_pairs = sum(len(nodes[k:]) for k in range(len(nodes)))
    done = 0
    k = 0
    for i in nodes:
        v_i = df_wide.get_column(i).to_numpy()
        for j in nodes[k:]:
            if progress is not None:
                progress(done, n_pairs, f"{i} / {j}")
            v_j = df_wide.get_column(j).to_numpy()
            valid = ~(np.isnan(v_i) | np.isnan(v_j))
            if np.sum(valid) < 3:
                xi_val = float("nan")
            else:
                x = v_i[valid]
                y = v_j[valid]
                ranks_y = rankdata(y)
                n = len(x)
                # Compute xi as 1 - (sum of squared rank differences) / (max possible)
                sorted_idx = np.argsort(x)
                d = np.sum(np.abs(np.diff(ranks_y[sorted_idx])))
                xi_val = abs(1.0 - (3 * d) / (n * n - 1))  # Normalize to [0, 1] via abs

            xi_values[i][j] = xi_val
            xi_values[j][i] = xi_val
            done += 1
        k += 1
    if progress is not None:
        progress(n_pairs, n_pairs, "done")
    return pl.DataFrame({row: [xi_values[row][col] for col in nodes] for row in nodes})


MEASURES: dict[str, Callable[..., pl.DataFrame]] = {
    "distance_correlation": distance_correlation_matrix,
    "pearson_correlation": pearson_correlation_matrix,
    "spearman_correlation": spearman_correlation_matrix,
    "maximal_correlation": maximal_correlation_matrix,
    "kendall_tau": kendall_tau_matrix,
    "dtw_distance": dtw_distance_matrix,
    "fastdtw_distance": fastdtw_distance_matrix,
    "shrinkage_correlation": shrinkage_correlation_matrix,
    "conditional_correlation": conditional_correlation_matrix,
    "mutual_information": mutual_information_matrix,
    "chatterjee_xi": chatterjee_xi_matrix,
}

MEASURE_LABELS: dict[str, str] = {
    "distance_correlation": "Distance correlation",
    "pearson_correlation": "Pearson correlation",
    "spearman_correlation": "Spearman correlation",
    "maximal_correlation": "Maximal correlation (ACE)",
    "kendall_tau": "Kendall Tau",
    "dtw_distance": "DTW distance",
    "fastdtw_distance": "FastDTW distance (approximate, faster)",
    "shrinkage_correlation": "Shrinkage correlation (Ledoit-Wolf)",
    "conditional_correlation": "Conditional correlation (stress regime)",
    "mutual_information": "Mutual information",
    "chatterjee_xi": "Chatterjee ξ correlation",
}

# Short tag for progress-bar/log display (e.g. "dcor 100% (780/780)").
MEASURE_SHORT_LABELS: dict[str, str] = {
    "distance_correlation": "dcor",
    "pearson_correlation": "pearson",
    "spearman_correlation": "spearman",
    "maximal_correlation": "ace",
    "kendall_tau": "kendall",
    "dtw_distance": "dtw",
    "fastdtw_distance": "fastdtw",
    "shrinkage_correlation": "shrinkage",
    "conditional_correlation": "conditional",
    "mutual_information": "mi",
    "chatterjee_xi": "xi",
}

# Measures gated behind a runtime-checked, optionally-compiled dependency.
# Excluded from available_measures() when that dependency isn't importable.
_MEASURE_REQUIRES_ACE = {"maximal_correlation"}

# Both DTW variants are API-only. Raw DTW is O(n^2) per pair with no
# windowing; FastDTW's Sakoe-Chiba band (see fastdtw_distance_matrix, tuned via
# Edge Settings' "FastDTW Settings" group) cuts that to O(n*radius), but it is
# still a pure-Python nested loop rather than a vectorised measure, so on a
# real panel it is far slower than every other option in the dropdown. Both
# stay reachable through compute_measure() for programmatic use, where the
# speed/accuracy trade-off is made deliberately.
_MEASURE_GUI_DISABLED = {"dtw_distance", "fastdtw_distance"}


def available_measures() -> list[tuple[str, str]]:
    """Measures selectable in the GUI: excludes ACE if ace_cream isn't usable, DTW (performance)."""
    return [
        (key, MEASURE_LABELS.get(key, key))
        for key in MEASURES
        if (key not in _MEASURE_REQUIRES_ACE or ACE_AVAILABLE)
        and key not in _MEASURE_GUI_DISABLED
    ]


def measure_short_label(measure_id: str) -> str:
    """Short tag for a measure id, for compact progress/log display."""
    return MEASURE_SHORT_LABELS.get(measure_id, measure_id)


def compute_measure(
    measure_id: str,
    df_wide: pl.DataFrame,
    nodes: list[str],
    progress: ProgressCallback | None = None,
    edge_settings: dict | None = None,
) -> pl.DataFrame:
    """Compute connection measure with optional edge settings.

    Args:
        measure_id: ID of the measure (e.g., 'spearman_correlation').
        df_wide: Wide-format DataFrame (rows=dates, cols=nodes).
        nodes: List of node names.
        progress: Optional progress callback.
        edge_settings: Optional dict with measure-specific parameters:
            - 'conditional_quantile': quantile threshold for conditional correlation (default 0.90)

    Returns:
        Square measure matrix as pl.DataFrame.
    """
    fn = MEASURES.get(measure_id)
    if fn is None:
        raise ValueError(f"Unknown measure: {measure_id!r}")

    edge_settings = edge_settings or {}

    # Pass edge settings to measures that support them
    if measure_id == "conditional_correlation":
        quantile = edge_settings.get("conditional_quantile", 0.90)
        return fn(df_wide, nodes, progress=progress, quantile=quantile)
    elif measure_id == "fastdtw_distance":
        radius = edge_settings.get("fastdtw_radius", 2)
        return fn(df_wide, nodes, progress=progress, radius=radius)
    else:
        return fn(df_wide, nodes, progress=progress)
