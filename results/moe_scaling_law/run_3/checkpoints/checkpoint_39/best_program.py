import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Power‐law scaling model for MoE architectures:
      L(Ne, D) = (a * Ne^alpha + b * D^beta)^(-p) + c

    Inputs:
      data_points: array-like of shape (N,2), columns [num_experts, dense_parameter_count]
      params:      array-like of 6 parameters [a, alpha, b, beta, p, c]

    Returns:
      (N,) array of predicted validation losses.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    # Ensure strictly positive inputs for powers
    Ne = np.clip(X[:, 0], 1.0, None)
    D  = np.clip(X[:, 1], 1.0, None)

    a, alpha, b, beta, p, c = params
    # Combined capacity term
    cap = a * (Ne ** alpha) + b * (D ** beta)
    cap = np.clip(cap, 1e-12, None)  # avoid zero or negative
    # Inverted power law plus offset
    return cap**(-p) + c


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6‐parameter scaling law by robust nonlinear least squares
    in log-loss space to reduce sensitivity to scale and outliers.

    Inputs:
      data_points: array-like of shape (N,2), [num_experts, dense_parameter_count]
      loss_values: array-like of shape (N,), observed validation losses

    Returns:
      params_opt: (6,) array of fitted parameters [a, alpha, b, beta, p, c]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()

    # Clip to avoid log(0)
    y = np.clip(y, 1e-12, None)

    # Initial guesses via medians
    Ne = X[:, 0]
    D  = X[:, 1]
    Ne_med = np.median(np.clip(Ne, 1.0, None))
    D_med  = np.median(np.clip(D,  1.0, None))
    y_min  = np.min(y)

    a0     = 1.0 / max(Ne_med, 1e-6)
    b0     = 1.0 / max(D_med,  1e-6)
    alpha0 = 0.5
    beta0  = 0.5
    p0     = 0.5
    c0     = max(0.0, y_min * 0.1)

    init_params = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)

    # Define residuals in log-space for relative-error fitting
    def residuals(params):
        pred = scaling_law_func(X, params)
        return np.log(pred) - np.log(y)

    # Bounds: scales ≥ 0, exponents limited to [0,5]
    lower_bounds = [0.0, 0.0,    0.0, 0.0,    0.0, 0.0]
    upper_bounds = [np.inf, 5.0, np.inf, 5.0,  5.0, np.inf]

    # Robust least-squares with soft_l1 loss in log-space
    result = least_squares(
        residuals,
        x0=init_params,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        loss='soft_l1',
        f_scale=0.1,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=20000
    )

    return result.x
# EVOLVE-BLOCK-END