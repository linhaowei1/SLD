# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predict validation loss using a combined power-law:
        L = (a * Ne^alpha + b * D^beta)^(-p) + c
    where:
        Ne = num_experts
        D  = dense_parameter_count
    params = [a, alpha, b, beta, p, c]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    Ne = np.maximum(X[:, 0], 1.0)    # ensure positivity
    D  = np.maximum(X[:, 1], 1.0)
    
    a, alpha, b, beta, p, c = params
    # combined capacity term
    cap = a * (Ne ** alpha) + b * (D ** beta)
    cap = np.maximum(cap, 1e-12)
    # inverted power-law plus offset
    loss_pred = cap ** (-p) + c
    return loss_pred

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter scaling law by non-linear least squares.
    Returns params = [a, alpha, b, beta, p, c].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    y = np.asarray(loss_values, dtype=float).ravel()
    
    # derive robust initial guesses
    Ne = X[:, 0]; D = X[:, 1]
    Ne_med = np.median(Ne[Ne > 0])
    D_med  = np.median(D[D   > 0])
    y_min  = np.min(y)
    
    a0     = 1.0 / max(Ne_med, 1e-6)
    b0     = 1.0 / max(D_med, 1e-6)
    alpha0 = 0.5
    beta0  = 0.5
    p0     = 0.5
    c0     = max(0.0, y_min * 0.1)
    
    init_params = np.array([a0, alpha0, b0, beta0, p0, c0], dtype=float)
    
    # bounds to keep exponents and scales in a reasonable range
    lower_bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper_bounds = [np.inf, 5.0, np.inf, 5.0, 5.0, np.inf]
    
    def residuals(params):
        return scaling_law_func(X, params) - y
    
    result = least_squares(
        residuals,
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