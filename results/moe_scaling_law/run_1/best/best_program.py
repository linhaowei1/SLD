import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict validation loss with a 6‐parameter combined power‐law model:
      L = (a * E^alpha + b * P^beta)^(-p) + c
    where
      E = num_experts,
      P = dense_parameter_count,
    params = [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    # guard against zeros
    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)

    a, alpha, b, beta, p, c = params
    # combined capacity term
    cap = a * (E ** alpha) + b * (P ** beta)
    cap = np.maximum(cap, 1e-12)
    return cap ** (-p) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6‐parameter scaling law by weighted non‐linear least squares.
    We weight residuals by 1/(y+eps) to focus on relative error.
    Returns optimized [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("data_points and loss_values must have same length")

    # clip inputs and targets for stability
    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)
    y_clip = np.clip(y, 1e-8, None)

    # sensible initial guesses
    a0 = 1.0 / np.median(E)
    b0 = 1.0 / np.median(P)
    alpha0 = 0.5
    beta0 = 0.5
    p0 = 1.0
    c0 = np.min(y_clip) * 0.1

    init = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)

    # bounds: amplitudes >0, exponents in [0,3], offset >=0
    lower = [1e-12, 0.0, 1e-12, 0.0, 0.0, 0.0]
    upper = [np.inf, 3.0, np.inf, 3.0, 3.0, np.max(y_clip)]

    # residuals weighted by 1/(y+eps)
    def _res(params):
        pred = scaling_law_func(X, params)
        return (pred - y) / (y_clip + 1e-8)

    sol = least_squares(
        _res,
        x0=init,
        bounds=(lower, upper),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=5000
    )

    return sol.x
# EVOLVE-BLOCK-END