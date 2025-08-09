import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict validation loss for MoE architectures via a 6-parameter combined power law:
        Ne = number of experts
        D  = dense parameter count
    Model:
        capacity = a * Ne**alpha + b * D**beta
        loss_pred = capacity**(-p) + c

    Inputs:
      data_points: (N,2) array [[Ne, D], ...]
      params:      length-6 array [a, alpha, b, beta, p, c]
    Output:
      loss_pred: (N,) array of predicted validation losses
    """
    X = np.asarray(data_points, dtype=float)
    # split features
    Ne = X[:, 0]
    D  = X[:, 1]
    # ensure strictly positive to avoid invalid powers
    Ne = np.clip(Ne, 1.0, None)
    D  = np.clip(D, 1.0, None)

    # unpack parameters
    a, alpha, b, beta, p, c = params

    # combined capacity term
    cap = a * (Ne ** alpha) + b * (D ** beta)
    cap = np.clip(cap, 1e-12, None)

    # inverted power law plus offset
    loss_pred = cap ** (-p) + c
    return loss_pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law parameters [a, alpha, b, beta, p, c]
    by minimizing the squared error between predicted and actual losses.
    Uses a bounded Trust Region Reflective solver.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()

    # clamp target to avoid pathological zeros
    y = np.clip(y, 1e-8, None)

    # extract features for initialization
    Ne = np.clip(X[:, 0], 1.0, None)
    D  = np.clip(X[:, 1], 1.0, None)

    # sensible initial guesses
    # use median scales to balance contributions
    Ne_med = np.median(Ne)
    D_med  = np.median(D)
    y_min  = np.min(y)

    # a and b scale inversely with median feature
    a0      = 1.0 / max(Ne_med, 1e-6)
    b0      = 1.0 / max(D_med, 1e-6)
    alpha0  = 0.5
    beta0   = 0.5
    p0      = 0.5
    c0      = max(0.0, y_min * 0.1)

    init_params = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)

    # set bounds to enforce positivity and reasonable exponent ranges
    lower_bounds = [1e-12, 0.0, 1e-12, 0.0, 0.0, 0.0]
    upper_bounds = [np.inf, 5.0, np.inf, 5.0, 5.0, np.inf]

    # define residuals for least-squares
    def residuals(params):
        return scaling_law_func(X, params) - y

    # run the optimizer
    result = least_squares(
        fun=residuals,
        x0=init_params,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=10000
    )

    return result.x

# EVOLVE-BLOCK-END