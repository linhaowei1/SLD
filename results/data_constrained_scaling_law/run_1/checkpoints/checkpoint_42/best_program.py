import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict loss as:
      loss ≈ E + A * tokens^(−b) * params^(−c) * unique_tokens^(−d)

    Inputs:
      data_points: array‐like of shape (N,3) with columns [tokens, params, unique_tokens]
      params:      array of length 5 or shape (T,5):
                   [A, b, c, d, E], all non-negative.
    Returns:
      preds: shape (N,) if params is length‐5, or (N,T) if multiple parameter sets.
    """
    X = np.atleast_2d(data_points).astype(float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    # avoid zero‐division / zero^exponent issues
    X = np.maximum(X, 1e-12)

    p = np.asarray(params, dtype=float)
    # Promote to 2D: (T,5)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    if p.shape[1] != 5:
        raise ValueError(f"Expected 5 parameters [A,b,c,d,E], got {p.shape[1]}")

    A = p[:, 0]  # shape (T,)
    b = p[:, 1]
    c = p[:, 2]
    d = p[:, 3]
    E = p[:, 4]

    # Compute term = A * tokens^(-b) * params^(-c) * unique_tokens^(-d)
    # We'll broadcast over N×T
    t_pow = X[:, [0]] ** (-b[np.newaxis, :])
    p_pow = X[:, [1]] ** (-c[np.newaxis, :])
    u_pow = X[:, [2]] ** (-d[np.newaxis, :])
    term = A[np.newaxis, :] * t_pow * p_pow * u_pow

    pred = E[np.newaxis, :] + term  # shape (N,T)
    # If only one parameter set, return shape (N,)
    return pred[:, 0] if pred.shape[1] == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 5‐parameter scaling law loss ≈ E + A·tokens^(−b)·params^(−c)·unique_tokens^(−d)
    by minimizing relative‐squared error in the loss domain:
      objective = mean(((pred - y) / (y + eps))^2)

    Returns:
      params_opt: array of length 5 [A, b, c, d, E], all non‐negative.
    """
    X = np.atleast_2d(data_points).astype(float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"data_points must have shape (N,3), got {X.shape}")
    y = np.asarray(loss_values, dtype=float).ravel()

    # Filter to finite, positive targets
    mask = np.isfinite(y) & (y > 0)
    X = X[mask]
    y = y[mask]
    N = y.shape[0]
    if N == 0:
        # fallback to trivial parameters
        return np.array([1.0, 0.0, 0.0, 0.0, np.mean(loss_values)], dtype=float)

    # Prepare logs for initial regression
    X_log = np.log(np.maximum(X, 1e-12))
    y_log = np.log(y + 1e-12)

    # Linear regression: y_log ≈ alpha + beta1·log(tokens) + beta2·log(params) + beta3·log(unique)
    D = np.hstack([np.ones((N, 1)), X_log])  # shape (N,4)
    beta, *_ = np.linalg.lstsq(D, y_log, rcond=None)
    alpha, b_lin, c_lin, d_lin = beta
    # Convert slopes to exponents b,c,d (we want loss ∼ A·tokens^(−b) ...):
    A0 = np.exp(alpha)
    b0 = max(-b_lin, 0.0)
    c0 = max(-c_lin, 0.0)
    d0 = max(-d_lin, 0.0)
    # Floor initial guess at fraction of minimum loss
    y_min = y.min()
    E0 = max(0.1 * y_min, 1e-6)

    x0 = np.array([A0, b0, c0, d0, E0], dtype=float)

    # Enforce non-negativity
    bounds = [(1e-12, None),  # A ≥ 0
              (0.0, None),    # b ≥ 0
              (0.0, None),    # c ≥ 0
              (0.0, None),    # d ≥ 0
              (0.0, None)]    # E ≥ 0

    eps = 1e-12

    def objective(p_flat):
        pred = scaling_law_func(X, p_flat)
        # relative squared error
        rel_err2 = ((pred - y) / (y + eps))**2
        return np.mean(rel_err2)

    res = minimize(objective,
                   x0,
                   method='L-BFGS-B',
                   bounds=bounds,
                   options={'ftol':1e-12, 'maxiter':5000})
    if res.success:
        return res.x
    else:
        return x0
# EVOLVE-BLOCK-END