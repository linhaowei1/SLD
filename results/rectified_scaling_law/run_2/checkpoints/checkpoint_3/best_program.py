import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts loss as a 4-parameter shifted power law:
      L(D) = c + A * (D + d0)^(-alpha)
    where:
      c     = asymptotic minimum loss
      A     = amplitude
      alpha = decay exponent (>0)
      d0    = data-size shift to improve fit stability (>=0)
    Inputs:
      data_points: array-like of shape (N,1) or (N,) with data sizes
      params:      array-like of 4 parameters [c, A, alpha, d0]
    Returns:
      preds: array of length N with predicted losses
    """
    D = np.asarray(data_points).ravel().astype(float)
    c, A, alpha, d0 = params
    # ensure non-negative inside power
    return c + A * np.power(np.maximum(D + d0, 1e-8), -alpha)

def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4-parameter scaling law to data via bounded least-squares.
    Returns optimized params = [c, A, alpha, d0].
    """
    D = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Initial guesses based on data heuristics
    c0     = max(0.0, np.min(y) * 0.9)
    A0     = np.max(y) - np.min(y)
    alpha0 = 0.5
    d0_0   = max(1.0, np.min(D) * 0.5)
    x0 = np.array([c0, A0, alpha0, d0_0], dtype=float)

    # Bounds to keep parameters in reasonable ranges
    lower = [0.0,       0.0,      0.0,    0.0]
    upper = [np.max(y)*2, (np.max(y)-np.min(y))*10 + 1.0, 10.0, np.max(D)*5]

    # Residuals: model prediction minus observed loss
    def resid(p):
        return scaling_law_func(D, p) - y

    # Jacobian of residuals for faster convergence
    def jac(p):
        c, A, alpha, d0 = p
        D_shift = np.maximum(D + d0, 1e-8)
        denom    = D_shift**(-alpha)
        log_term = np.log(D_shift)
        # derivatives of L w.r.t [c, A, alpha, d0]
        d_dc     = np.ones_like(D)
        d_dA     = denom
        d_dalpha = -A * log_term * denom
        d_dd0    = -A * alpha * D_shift**(-alpha - 1)
        # residual = pred - y, so same jacobian
        return np.vstack((d_dc, d_dA, d_dalpha, d_dd0)).T

    # Perform bounded least-squares optimization
    result = least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lower, upper),
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=2000,
    )

    # Return the best-fit parameters [c, A, alpha, d0]
    return result.x