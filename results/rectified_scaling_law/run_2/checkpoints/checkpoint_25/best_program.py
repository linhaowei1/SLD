# EVOLVE-BLOCK-START
"""
Improved 4-parameter shifted power law with log-domain residuals
and robust fitting to emphasize relative errors.

Model:
  L(D) = c + A * (D + d0)^(-alpha)

We fit parameters p = [c, A, alpha, d0] by minimizing
  sum_i ρ( log(L(D_i; p)) - log(y_i) )
with a soft L1 loss ρ, to balance absolute and relative errors.
"""
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(data_points, params):
    """
    Predicts loss for data sizes using a shifted power law.
    Inputs:
      data_points: array-like, shape (N,1) or (N,)
      params:      length-4 array [c, A, alpha, d0]
    Outputs:
      preds:       array of shape (N,)
    """
    D = np.asarray(data_points).ravel().astype(float)
    c, A, alpha, d0 = params
    # small epsilon for numerical stability
    eps = 1e-12
    D_shift = np.maximum(D + d0, eps)
    return c + A * D_shift**(-alpha)

def fit_scaling_law(data_points, loss_values):
    """
    Fits the 4-parameter scaling law by robust log-domain least-squares.
    Returns:
      params: array [c, A, alpha, d0]
    """
    D = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)

    # Heuristic initial guesses
    c0     = max(0.0, 0.9 * np.min(y))
    A0     = np.ptp(y)                    # max(y) - min(y)
    alpha0 = 0.5
    d0_0   = max(0.0, 0.1 * np.min(D))
    x0 = np.array([c0, A0, alpha0, d0_0], dtype=float)

    # Bounds for parameters to ensure stability
    lower = [0.0,                0.0,   0.0,       0.0]
    upper = [np.min(y),  10.0 * A0 + 1.0,   5.0,   np.max(D)]

    # Residuals in log-domain: log(model) - log(observed)
    def resid(p):
        pred = scaling_law_func(D, p)
        return np.log(pred) - np.log(y)

    # Analytic Jacobian of the log-residuals
    def jac(p):
        c, A, alpha, d0 = p
        eps = 1e-12
        D_shift = np.maximum(D + d0, eps)
        denom   = D_shift**(-alpha)
        pred     = c + A * denom
        # partial derivatives of L wrt p
        dL_dc     = np.ones_like(D)
        dL_dA     = denom
        dL_dalpha = -A * np.log(D_shift) * denom
        dL_dd0    = -A * alpha * D_shift**(-alpha - 1)
        # Jacobian of log-residual = (1 / pred) * dL/dp
        inv_pred = 1.0 / pred
        return np.vstack([
            dL_dc * inv_pred,
            dL_dA * inv_pred,
            dL_dalpha * inv_pred,
            dL_dd0 * inv_pred
        ]).T

    # Solve with a soft-L1 robust loss to balance absolute & relative errors
    result = least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lower, upper),
        loss='soft_l1',
        f_scale=0.1,
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=2000
    )
    return result.x
# EVOLVE-BLOCK-END