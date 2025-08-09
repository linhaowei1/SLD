# EVOLVE-BLOCK-START
"""
Scaling law discovery for MoE LLM finetuning scenarios
Enhanced 6-parameter combined‐capacity power‐law model with offset:
    L(N_e, D) = (a * N_e^alpha + b * D^beta)^(-p) + c
Parameters: [a, alpha, b, beta, p, c] (6 total)
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict validation loss given expert count and dense parameter count.
    Model: L = (a * Ne^alpha + b * D^beta)^(-p) + c

    Inputs:
      data_points: array of shape (N,2) with columns [num_experts, dense_parameter_count]
      params: length-6 array [a, alpha, b, beta, p, c]

    Returns:
      loss_pred: array of shape (N,) of predicted validation losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    Ne = np.clip(X[:, 0], 1e-8, None)
    D  = np.clip(X[:, 1], 1e-8, None)

    a, alpha, b, beta, p, c = params

    # combined capacity term
    cap = a * (Ne**alpha) + b * (D**beta)
    cap = np.clip(cap, 1e-12, None)

    # inverted power-law plus offset
    loss_pred = cap**(-p) + c
    return loss_pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter combined-capacity scaling law to data.
    Returns optimized params [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()

    # ensure positive features
    Ne = np.clip(X[:, 0], 1e-8, None)
    D  = np.clip(X[:, 1], 1e-8, None)

    # derive robust initial guesses
    y_min = np.min(y)
    Ne_med = np.median(Ne)
    D_med  = np.median(D)

    # a and b such that a*Ne_med^alpha ≈ b*D_med^beta ≈ 1/y_min
    # start with alpha=beta=0.5, p=1
    alpha0, beta0, p0 = 0.5, 0.5, 1.0
    a0 = (1.0 / max(y_min, 1e-8)) / (Ne_med**alpha0)
    b0 = (1.0 / max(y_min, 1e-8)) / (D_med**beta0)
    c0 = max(0.0, y_min * 0.1)

    init_params = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)

    # bounds: scales ≥0, exponents in [0,5], offset ≥0
    lower_bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper_bounds = [np.inf, 5.0, np.inf, 5.0, 5.0, np.inf]

    def residuals(params):
        return scaling_law_func(X, params) - y

    # robust least-squares fitting with soft_l1 loss to dampen outliers
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