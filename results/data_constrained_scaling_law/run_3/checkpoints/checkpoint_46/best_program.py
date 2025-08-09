import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict loss as an additive combination of three decaying power‐law terms
    plus a floor:

      loss ≈ C0
             + A_t * tokens^(−α_t)
             + A_p * params^(−α_p)
             + A_u * unique_tokens^(−α_u)

    Inputs:
      data_points: array‐like of shape (N,3) [tokens, params, unique_tokens]
      params:      array‐like of length 7:
                   [C0,
                    A_t, α_t,
                    A_p, α_p,
                    A_u, α_u]
    Returns:
      preds: np.ndarray of shape (N,) with predicted loss
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 7:
        raise ValueError(f"Expected 7 parameters [C0, A_t, α_t, A_p, α_p, A_u, α_u], got {p.size}")
    C0, A_t, α_t, A_p, α_p, A_u, α_u = p

    # numeric floor to avoid zero or negative bases
    eps = 1e-12
    tokens = np.maximum(X[:, 0], eps)
    prms   = np.maximum(X[:, 1], eps)
    uniqs  = np.maximum(X[:, 2], eps)

    # additive power‐law model
    preds = (
        C0
        + A_t * tokens**(-α_t)
        + A_p * prms**(-α_p)
        + A_u * uniqs**(-α_u)
    )
    return preds

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7 parameters of the additive power‐law scaling law by minimizing
    relative error residuals via trust‐region least squares.

    Returns:
      params_opt: np.ndarray of length 7
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    y = np.asarray(loss_values, dtype=float).ravel()

    # ensure positive losses for relative residual
    eps_y = 1e-12
    y = np.maximum(y, eps_y)

    # initialize floor C0 slightly below min observed loss
    y_min = np.min(y)
    C0_0  = max(0.0, 0.9 * y_min)

    # distribute remaining mean loss equally as initial amplitudes
    mean_resid = np.mean(y - C0_0)
    amp0 = max(mean_resid / 3.0, eps_y)

    # initial decay exponents
    dec0 = 0.5

    # initial parameter vector: [C0, A_t, α_t, A_p, α_p, A_u, α_u]
    p0 = np.array([
        C0_0,
        amp0, dec0,
        amp0, dec0,
        amp0, dec0
    ], dtype=float)

    # parameter bounds: floor ≥ 0, amplitudes ≥ tiny positive, decays ≥ 0
    lower = np.array([0.0, eps_y, 0.0, eps_y, 0.0, eps_y, 0.0], dtype=float)
    upper = np.full(7, np.inf, dtype=float)

    # residual function (relative)
    def resid_fn(p):
        pred = scaling_law_func(X, p)
        return (pred - y) / y

    sol = least_squares(
        resid_fn,
        p0,
        bounds=(lower, upper),
        method='trf',
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=10000
    )

    # return optimized parameters or fallback to initial
    return sol.x if sol.success else p0
# EVOLVE-BLOCK-END