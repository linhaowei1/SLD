# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts loss as a 4-parameter shifted power law:
      L(N) = B + A * (N + C)^(-alpha)
    where all parameters B, A, C, alpha are constrained >= 0.
    Inputs:
      data_points: array-like of shape (N,) or (N,1) with data sizes
      params:      array-like of 4 parameters [B, A, C, alpha]
    Returns:
      preds: array of length N with predicted losses
    """
    N = np.asarray(data_points).ravel().astype(float)
    B, A, C, alpha = params
    # Evaluate the shifted power law
    # (N + C) > 0 since C >= 0 and N > 0 in our setting
    return B + A * np.power(N + C, -alpha)

def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4-parameter scaling law L(N) = B + A*(N + C)^(-alpha)
    to the provided (data_points, loss_values) via bounded least-squares.
    Returns optimized params [B, A, C, alpha].
    """
    X = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Heuristic initial guesses
    B0     = max(0.0, np.min(y))                   # asymptotic minimum loss
    A0     = max(1e-6, np.max(y) - np.min(y))       # amplitude of decay
    C0     = max(0.0, np.min(X) * 0.5)              # horizontal shift
    alpha0 = 0.5                                     # decay exponent
    x0 = np.array([B0, A0, C0, alpha0], dtype=float)

    # Bounds: all parameters non-negative
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.inf, np.inf, np.inf, np.inf]

    # Residual function
    def residuals(p):
        return scaling_law_func(X, p) - y

    # Solve with bounded least-squares for stability
    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        max_nfev=2000
    )

    return result.x
# EVOLVE-BLOCK-END