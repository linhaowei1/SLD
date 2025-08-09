import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    7-parameter additive power-law scaling law with a floor:
      loss ≈ C0
             + A_t · tokens^(−α_t)
             + A_p · params^(−α_p)
             + A_u · unique_tokens^(−α_u)

    Inputs:
      data_points: array-like, shape (N,3) columns [tokens, params, unique_tokens]
      params:      length-7 array [C0,
                                   A_t, α_t,
                                   A_p, α_p,
                                   A_u, α_u]

    Returns:
      preds: np.ndarray of shape (N,) with predicted loss values
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")

    p = np.asarray(params, dtype=float).ravel()
    if p.size != 7:
        raise ValueError(f"Expected 7 parameters [C0, A_t, α_t, A_p, α_p, A_u, α_u], got {p.size}")
    C0, A_t, alpha_t, A_p, alpha_p, A_u, alpha_u = p

    # avoid zeros/negatives in power computations
    eps = 1e-12
    tokens = np.maximum(X[:, 0], eps)
    params_count = np.maximum(X[:, 1], eps)
    uniqs = np.maximum(X[:, 2], eps)

    # compute additive power-law contributions plus floor
    preds = (
        C0
        + A_t * np.power(tokens, -alpha_t)
        + A_p * np.power(params_count, -alpha_p)
        + A_u * np.power(uniqs,     -alpha_u)
    )
    return preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7 parameters of the additive power-law scaling law by minimizing
    relative errors via a trust-region least-squares solver.

    Returns:
      params_opt: np.ndarray of length 7
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    y = np.asarray(loss_values, dtype=float).ravel()

    # ensure positive targets for relative residuals
    eps = 1e-12
    y = np.maximum(y, eps)

    # initial floor guess: 90% of the minimum observed loss
    C0_0 = max(0.0, 0.9 * np.min(y))

    # distribute remaining mean loss equally as initial amplitudes
    rem = np.mean(y - C0_0)
    A0 = max(rem / 3.0, eps)

    # initial decay exponents
    decay0 = 0.5

    # initial parameter vector: [C0, A_t, α_t, A_p, α_p, A_u, α_u]
    p0 = np.array([C0_0,
                   A0,      decay0,
                   A0,      decay0,
                   A0,      decay0], dtype=float)

    # bounds: floor ≥ 0; amplitudes ≥ tiny positive; decays ≥ 0
    lower = np.array([0.0,
                      eps,    0.0,
                      eps,    0.0,
                      eps,    0.0], dtype=float)
    upper = np.full(7, np.inf, dtype=float)

    # residual function: normalized by target to balance scales
    def resid_fn(p):
        pred = scaling_law_func(X, p)
        return (pred - y) / (y + eps)

    # solve with Trust-Region-Reflective least squares
    sol = least_squares(
        resid_fn,
        p0,
        bounds=(lower, upper),
        method='trf',
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=10000
    )

    return sol.x if sol.success else p0

# EVOLVE-BLOCK-END