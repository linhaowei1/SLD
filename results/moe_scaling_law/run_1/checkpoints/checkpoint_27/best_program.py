# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict validation loss via a 6‐parameter combined power‐law:
      L = (a * Ne^alpha + b * D^beta)^(-p) + c
    where:
      Ne = num_experts (≥1),
      D  = dense_parameter_count (≥1),
    params = [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    # enforce positivity for features
    Ne = np.clip(X[:, 0], 1.0, None)
    D  = np.clip(X[:, 1], 1.0, None)

    a, alpha, b, beta, p, c = params
    # combined capacity term
    cap = a * (Ne ** alpha) + b * (D ** beta)
    # avoid zero or negative
    cap = np.clip(cap, 1e-12, None)
    # inverted power‐law plus offset
    return cap ** (-p) + c

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6‐parameter scaling law by robust non‐linear least squares.
    Uses a soft_l1 loss to mitigate outliers.
    Returns optimized params = [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()

    # ensure valid
    y = np.clip(y, 1e-12, None)
    Ne = np.clip(X[:, 0], 1e-8, None)
    D  = np.clip(X[:, 1], 1e-8, None)

    # medians for initialization
    Ne_med = np.median(Ne)
    D_med  = np.median(D)
    y_min  = np.min(y)

    # initial exponent guesses
    alpha0 = 0.5
    beta0  = 0.5
    p0     = 0.5
    # initial scale guesses from approximate inversion: L ≈ (a Ne^α)^(-p) => a ≈ (L^(-1/p)) / Ne^α
    a0 = max((y_min ** (-1.0 / p0)) / (Ne_med ** alpha0), 1e-6)
    b0 = max((y_min ** (-1.0 / p0)) / (D_med ** beta0), 1e-6)
    # small offset
    c0 = max(0.0, y_min * 0.05)

    init_params = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)

    # bounds for stability: scales and exponents positive, c between 0 and min(y)
    lower_bounds = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 0.0]
    upper_bounds = [np.inf, 5.0, np.inf, 5.0, 5.0, y_min]

    def residuals(params):
        return scaling_law_func(X, params) - y

    result = least_squares(
        residuals,
        x0=init_params,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=5000
    )

    return result.x
# EVOLVE-BLOCK-END