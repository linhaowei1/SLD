# EVOLVE-BLOCK-START
"""
Refined 4-parameter scaling-law for LLM fine-tuning:
    L(N) = B + A * (N + C)^(-α)
Parameters (A, α, C, B) are enforced positive via exp‐reparameterization
(p = [log A, log α, log C, log B]). We use L-BFGS-B with sensible bounds
and data-driven initial guesses for robustness and stability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict loss given data sizes and log-domain parameters.
    Inputs:
      data_points: array-like of shape (N,) or (N,1) with data sizes
      params:      array-like of 4 entries [pA, pα, pC, pB] in log-domain
    Returns:
      preds:       ndarray of shape (N,) with predicted losses
    """
    X = np.asarray(data_points).ravel().astype(float)
    p = np.asarray(params).ravel()
    if p.size != 4:
        raise ValueError(f"scaling_law_func expects 4 parameters, got {p.size}")
    # Reconstruct positive-valued parameters
    A     = np.exp(p[0])    # amplitude
    alpha = np.exp(p[1])    # decay exponent
    C     = np.exp(p[2])    # horizontal shift
    B     = np.exp(p[3])    # asymptotic floor
    # Compute prediction
    return B + A * np.power(X + C, -alpha)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 4-parameter scaling law:
      L(N) = B + A * (N + C)^(-α)
    Returns optimized log-domain parameters [log A, log α, log C, log B].
    """
    # Prepare 1D float arrays
    X = np.asarray(data_points).ravel().astype(float)
    y = np.asarray(loss_values).ravel().astype(float)
    # Sort by X for stable slope estimation
    idx = np.argsort(X)
    Xs, ys = X[idx], y[idx]
    y_min, y_max = ys.min(), ys.max()

    # Initial guess: amplitude ~ range, floor ~ 0.9*min, shift ~ geometric mean
    A0 = max(y_max - y_min, 1e-2)
    B0 = max(0.9 * y_min, 1e-3)
    # geometric mean for shift
    C0 = np.exp(np.mean(np.log(np.clip(Xs, 1e-8, None))))
    # estimate exponent α from endpoint slope in log–log space
    if ys[0] > B0 and ys[-1] > B0 and Xs[0] != Xs[-1]:
        y_adj0 = ys[0] - B0
        y_adj1 = ys[-1] - B0
        try:
            slope = - (np.log(y_adj1) - np.log(y_adj0)) / (np.log(Xs[-1] + C0) - np.log(Xs[0] + C0))
            alpha0 = float(np.clip(slope, 1e-3, 10.0))
        except Exception:
            alpha0 = 0.5
    else:
        alpha0 = 0.5

    # Pack initial log-domain parameters
    p0 = np.log([A0, alpha0, C0, B0])

    # Objective: mean squared error
    def _mse(p):
        pred = scaling_law_func(X, p)
        return np.mean((pred - y) ** 2)

    # Bounds on log-parameters to avoid extreme values
    bounds = [(-20, 20),   # log A
              (-5,  5),    # log α
              (-20, 20),   # log C
              (-20, 20)]   # log B

    # Optimize with L-BFGS-B
    result = minimize(
        _mse,
        p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 10000}
    )

    # Return optimized or fallback to initial
    return result.x if result.success else p0
# EVOLVE-BLOCK-END