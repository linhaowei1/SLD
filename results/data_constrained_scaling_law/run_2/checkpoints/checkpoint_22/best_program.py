import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict loss using a shifted power‐law scaling:
      loss = exp(log_a)
             * (tokens + t0)^(-b)
             * (params + p0)^(-c)
             * (unique_tokens + u0)^(-d)

    params: array of length 7 or shape (M,7):
        [log_a, b, c, d, k0, k1, k2]
      where t0 = exp(k0), p0 = exp(k1), u0 = exp(k2).
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")

    p = np.asarray(params, dtype=float)
    # Ensure p is 2D: (M,7)
    if p.ndim == 1:
        p_arr = p[np.newaxis, :]
    else:
        p_arr = p.copy()
    M, P = p_arr.shape
    if P != 7:
        raise ValueError(f"Expected 7 parameters, got {P}")

    tokens = X[:, 0]
    param_cnt = X[:, 1]
    uniq = X[:, 2]

    N = X.shape[0]
    preds = np.zeros((N, M), dtype=float)

    # Vectorized over M parameter sets
    for i in range(M):
        log_a, b, c, d, k0, k1, k2 = p_arr[i]
        t0 = np.exp(k0)
        p0 = np.exp(k1)
        u0 = np.exp(k2)

        # Compute log‐prediction
        y_log = (
            log_a
            - b * np.log(tokens + t0)
            - c * np.log(param_cnt + p0)
            - d * np.log(uniq + u0)
        )
        # Avoid numerical under/overflow
        y_log = np.clip(y_log, -50.0, 50.0)
        preds[:, i] = np.exp(y_log)

    # Return shape (N,) if single param set
    return preds[:, 0] if M == 1 else preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7‐parameter shifted‐power‐law scaling law by minimizing
    mean squared error in log‐loss space.
    Returns an array of 7 optimized parameters:
        [log_a, b, c, d, k0, k1, k2]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    y = np.asarray(loss_values, dtype=float).reshape(-1)

    # Keep only strictly positive losses
    mask = y > 0
    X_fit = X[mask]
    y_fit = y[mask]
    if X_fit.shape[0] == 0:
        raise ValueError("No positive loss values to fit.")

    # Work in log-loss space
    y_log = np.log(y_fit)

    # Initial guess:
    #   log_a = log(mean(loss))
    #   b,c,d = 0.5
    #   k0,k1,k2 = 0 => t0=p0=u0=1
    init = np.zeros(7, dtype=float)
    init[0] = np.log(np.mean(y_fit))
    init[1:4] = 0.5
    init[4:7] = 0.0

    # Bounds: exponents b,c,d >= 0; others unbounded
    bounds = [
        (None, None),  # log_a
        (0.0, None),   # b
        (0.0, None),   # c
        (0.0, None),   # d
        (None, None),  # k0
        (None, None),  # k1
        (None, None),  # k2
    ]

    # Objective: mean squared error in log‐space
    def mse_obj(p):
        preds = scaling_law_func(X_fit, p)
        # work in log
        eps = 1e-12
        y_pred_log = np.log(np.clip(preds, eps, None))
        resid = y_pred_log - y_log
        return np.mean(resid * resid)

    res = minimize(
        mse_obj,
        init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )
    if not res.success:
        # fallback to initial guess
        return init
    return res.x

# EVOLVE-BLOCK-END