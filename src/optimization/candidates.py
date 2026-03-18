"""Structured candidate generation from top-K surrogate-ranked designs.

After the surrogate ranks a large pool of candidates, this module generates
new designs by *combining* features of the best ones -- like a learned
crossover operator guided by the surrogate's ranking.
"""

import numpy as np
from sklearn.decomposition import PCA

# Feature group slices for 30 design variables (match design_variables.py order)
_PLANFORM = slice(0, 4)     # half_span, wing_root_chord, taper, sweep
_BODY = slice(4, 12)        # 8 body variables
_TWIST = slice(12, 17)      # 5 twist angles
_DIHEDRAL = slice(17, 22)   # 5 dihedral angles
_AIRFOIL = slice(22, 30)    # 8 Kulfan weights + deltas

_GROUPS = [_PLANFORM, _BODY, _TWIST, _DIHEDRAL, _AIRFOIL]


def generate_candidates(
    top_k: np.ndarray,
    n_candidates: int,
    bounds: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    feasible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Generate new design candidates from top-K surrogate-ranked designs.

    Combines up to 6 strategies: centroids, convex combinations, group-wise
    recombination, PCA-guided perturbation, boundary interpolation (if
    feasible_mask provided), and local perturbation.

    Parameters
    ----------
    top_k : ndarray, shape (K, 30)
        Best designs ranked by surrogate objective.
    n_candidates : int
        Total number of candidates to generate.
    bounds : tuple of (lb, ub) arrays, each shape (30,)
    rng : numpy random Generator
    feasible_mask : ndarray of bool, shape (K,), optional
        If provided, enables boundary interpolation between feasible and
        infeasible designs to explore the feasibility frontier.

    Returns
    -------
    ndarray, shape (n_candidates, 30)
    """
    lb, ub = bounds
    K, D = top_k.shape

    # Check if boundary interpolation is possible
    use_boundary = (
        feasible_mask is not None
        and len(feasible_mask) == K
        and feasible_mask.any()
        and not feasible_mask.all()
    )

    # Allocate counts per strategy
    n_centroid = max(2, int(0.10 * n_candidates))
    if use_boundary:
        n_convex = max(1, int(0.25 * n_candidates))
        n_group = max(1, int(0.15 * n_candidates))
        n_pca = max(1, int(0.15 * n_candidates))
        n_boundary = max(1, int(0.15 * n_candidates))
    else:
        n_convex = max(1, int(0.30 * n_candidates))
        n_group = max(1, int(0.20 * n_candidates))
        n_pca = max(1, int(0.20 * n_candidates))
        n_boundary = 0
    n_local = n_candidates - n_centroid - n_convex - n_group - n_pca - n_boundary

    parts = []

    # --- 1. Centroids (10%) ---
    parts.append(_centroids(top_k, n_centroid, rng))

    # --- 2. Convex combinations (25-30%) ---
    parts.append(_convex_combinations(top_k, n_convex, rng))

    # --- 3. Group-wise recombination (15-20%) ---
    parts.append(_group_recombination(top_k, n_group, rng))

    # --- 4. PCA-guided perturbation (15-20%) ---
    parts.append(_pca_perturbation(top_k, n_pca, rng))

    # --- 5. Boundary interpolation (15%, if feasible_mask available) ---
    if use_boundary and n_boundary > 0:
        parts.append(_boundary_interpolation(top_k, n_boundary, rng, feasible_mask))

    # --- 6. Local perturbation around best (remainder) ---
    parts.append(_local_perturbation(top_k, n_local, rng))

    candidates = np.vstack(parts)

    # Clip to bounds
    candidates = np.clip(candidates, lb, ub)

    return candidates[:n_candidates]


def _centroids(top_k: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Mean, median, and group-mixed centroids."""
    results = [
        np.mean(top_k, axis=0),
        np.median(top_k, axis=0),
    ]

    # Group-mixed: for each candidate, pick centroid type per group randomly
    for _ in range(n - 2):
        candidate = np.empty(top_k.shape[1])
        for grp in _GROUPS:
            if rng.random() < 0.5:
                candidate[grp] = np.mean(top_k[:, grp], axis=0)
            else:
                candidate[grp] = np.median(top_k[:, grp], axis=0)
        results.append(candidate)

    return np.array(results[:n])


def _convex_combinations(top_k: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random interpolation between pairs from top-K."""
    K = len(top_k)
    results = []
    for _ in range(n):
        i, j = rng.choice(K, size=2, replace=False)
        alpha = rng.random()
        results.append(alpha * top_k[i] + (1 - alpha) * top_k[j])
    return np.array(results)


def _group_recombination(top_k: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample each feature group independently from different top-K designs."""
    K = len(top_k)
    results = []
    for _ in range(n):
        candidate = np.empty(top_k.shape[1])
        for grp in _GROUPS:
            donor = rng.integers(K)
            candidate[grp] = top_k[donor, grp]
        results.append(candidate)
    return np.array(results)


def _pca_perturbation(top_k: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample in the PCA subspace of top-K designs."""
    K, D = top_k.shape

    if K < 3:
        # Not enough samples for PCA -- fall back to local perturbation
        return _local_perturbation(top_k, n, rng)

    center = np.mean(top_k, axis=0)
    n_components = min(K - 1, D, 10)  # cap at 10 components
    pca = PCA(n_components=n_components).fit(top_k)

    # Sample in PCA space with eigenvalue-scaled noise
    scores = rng.normal(0, 1, size=(n, n_components))
    scores *= np.sqrt(pca.explained_variance_)  # scale by eigenvalues

    results = center + scores @ pca.components_
    return results


def _boundary_interpolation(
    top_k: np.ndarray, n: int, rng: np.random.Generator,
    feasible_mask: np.ndarray,
) -> np.ndarray:
    """Interpolate between feasible and infeasible designs near the boundary."""
    feas_idx = np.where(feasible_mask)[0]
    infeas_idx = np.where(~feasible_mask)[0]

    results = []
    for _ in range(n):
        i = rng.choice(feas_idx)
        j = rng.choice(infeas_idx)
        # Interpolate in [0.3, 0.7] to stay near the boundary
        alpha = rng.uniform(0.3, 0.7)
        results.append(alpha * top_k[i] + (1 - alpha) * top_k[j])
    return np.array(results)


def _local_perturbation(top_k: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian perturbation around the best design, sigma from top-K spread."""
    best = top_k[0]  # top_k is ranked, index 0 = best

    # Adaptive sigma: std of top-K per feature, floored at 1% of range
    sigma = np.std(top_k, axis=0)
    sigma = np.maximum(sigma, 0.01 * np.ptp(top_k, axis=0))
    # If top-K is degenerate (all same), use 5% of best value
    sigma = np.where(sigma < 1e-10, 0.05 * np.abs(best) + 1e-6, sigma)

    results = best + rng.normal(0, 1, size=(n, len(best))) * sigma
    return results
