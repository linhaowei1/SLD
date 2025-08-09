import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Enhanced 7-parameter scaling law:
      Let f1 = log(tokens + eps), f2 = log(params + eps), f3 = log(unique_tokens + eps).
      Then
        log_loss_adj = a0
                     + a1*f1 + a2*f2 + a3*f3
                     + a4*(f1*f2) + a5*(f1*f3)
      and
        loss = C0 + exp(log_loss_adj)
    where params = [C0, a0, a1, a2, a3, a4, a5].
    """
    X = np.atleast_2d(data_points).astype(float)
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 7:
        raise ValueError(f"Expected 7 parameters [C0,a0..a5], got {p.size}")

    C0, a0, a1, a2, a3, a4, a5 = p
    eps = 1e-8
    f1 = np.log(X[:, 0] + eps)
    f2 = np.log(X[:, 1] + eps)
    f3 = np.log(X[:, 2] + eps)

    # build log-adjusted prediction
    log_loss_adj = (
        a0
        + a1 * f1
        + a2 * f2
        + a3 * f3
        + a4 * (f1 * f2)
        + a5 * (f1 * f3)
    )
    # clamp for numerical stability
    log_loss_adj = np.clip(log_loss_adj, -50.0, 50.0)
    return C0 + np.exp(log_loss_adj)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7-parameter cross-interaction scaling law:
      params = [C0, a0, a1, a2, a3, a4, a5]
    by minimizing MSE in the log-domain (after subtracting floor).
    """
    X = np.atleast_2d(data_points).astype(float)
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    y = np.asarray(loss_values, dtype=float).ravel()
    N = y.shape[0]
    if N == 0:
        return np.zeros(7, dtype=float)

    # initialize C0 as small fraction of min(loss)
    y_min = np.min(y)
    C0_init = max(0.0, 0.1 * y_min)

    # Construct features for initial linear fit in log-domain
    eps = 1e-8
    mask = (y > C0_init + eps)
    if mask.sum() < 3:
        # fallback to simple power-law fit if too few points
        C0_init = 0.0
        mask = (y > eps)
    Xf = X[mask]
    yf = y[mask] - C0_init

    f1 = np.log(Xf[:, 0] + eps)
    f2 = np.log(Xf[:, 1] + eps)
    f3 = np.log(Xf[:, 2] + eps)
    y_log = np.log(yf + eps)

    # design matrix: [1, f1, f2, f3, f1*f2, f1*f3]
    D = np.column_stack([
        np.ones_like(f1),
        f1, f2, f3,
        f1 * f2, f1 * f3
    ])
    # ridge-regression initialization
    reg = 1e-6
    G = D.T.dot(D) + reg * np.eye(D.shape[1])
    rhs = D.T.dot(y_log)
    try:
        a_init = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        a_init, *_ = np.linalg.lstsq(D, y_log, rcond=None)

    # initial parameter vector
    p0 = np.concatenate([[C0_init], a_init])

    # bounds: C0 ≥ 0, others unbounded
    bounds = [(0.0, None)] + [(-np.inf, np.inf)] * 6

    def obj(p):
        # MSE in log-domain after subtracting floor
        C0 = p[0]
        y_pred = scaling_law_func(X, p)
        # ensure positivity
        y_adj = np.maximum(y - C0, eps)
        y_pred_adj = np.maximum(y_pred - C0, eps)
        resid = np.log(y_pred_adj) - np.log(y_adj)
        return np.mean(resid * resid)

    res = minimize(
        obj,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 1000}
    )
    return res.x if res.success else p0

# EVOLVE-BLOCK-END