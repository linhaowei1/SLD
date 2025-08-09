import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict loss as an additive combination of three decaying power‐law terms
    plus a floor:

      loss ≈ L_inf + A_t·tokens^(−α_t) + A_p·params^(−α_p) + A_u·unique_tokens^(−α_u)

    We parametrize all seven quantities in log‐space to enforce positivity:
      [u0..u6] → exp(ui) for each.

    Inputs:
      data_points: array‐like (N,3) columns [tokens, params, unique_tokens]
      params:      length‐7 array or shape (T,7) of raw log‐parameters
                   [u0,u1,u2,u3,u4,u5,u6]

    Returns:
      preds: shape (N,) if single param‐vector or (N,T) if multiple
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    # avoid zeros
    eps = 1e-12
    tkn = np.maximum(X[:, 0], eps)
    prm = np.maximum(X[:, 1], eps)
    ut  = np.maximum(X[:, 2], eps)

    # ensure params is 2D: (T,7)
    p = np.asarray(params, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    T, P = p.shape
    if P != 7:
        raise ValueError("params must have length 7 (got %d)" % P)

    # unpack raw parameters
    u0 = p[:, 0]  # floor
    u1 = p[:, 1]; u2 = p[:, 2]  # tokens amplitude & decay
    u3 = p[:, 3]; u4 = p[:, 4]  # params amplitude & decay
    u5 = p[:, 5]; u6 = p[:, 6]  # unique_tokens amplitude & decay

    # map to positive
    L_inf   = np.exp(u0)[None, :]      # (1,T)
    A_t     = np.exp(u1)[None, :]
    alpha_t = np.exp(u2)[None, :]
    A_p     = np.exp(u3)[None, :]
    alpha_p = np.exp(u4)[None, :]
    A_u     = np.exp(u5)[None, :]
    alpha_u = np.exp(u6)[None, :]

    # compute predictions (N,T)
    preds = (
        L_inf
        + A_t * (tkn[:, None] ** (-alpha_t))
        + A_p * (prm[:, None] ** (-alpha_p))
        + A_u * (ut[:, None] ** (-alpha_u))
    )

    # return (N,) if single parameter‐vector
    return preds[:, 0] if T == 1 else preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7 raw log‐parameters [u0..u6] by minimizing squared error:

      resid(u) = scaling_law_func(X, u) – y

    We initialize
      u0 = log(min(y)),           floor
      u1,u3,u5 = log((mean(y)−floor)/3), amplitudes
      u2,u4,u6 = log(1.0),         decays
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    y = np.asarray(loss_values, dtype=float).reshape(-1)

    # ignore non‐positive losses
    mask = y > 0
    X = X[mask]
    y = y[mask]

    # statistics for initialization
    eps = 1e-12
    mean_y = np.mean(y)
    min_y  = np.min(y)
    u0 = np.log(min_y + eps)                        # floor
    resid = max(mean_y - np.exp(u0), eps)
    amp0 = resid / 3.0
    u1 = np.log(amp0 + eps); u3 = u1; u5 = u1       # initial amplitudes
    u2 = np.log(1.0); u4 = np.log(1.0); u6 = np.log(1.0)  # initial decays

    init = np.array([u0, u1, u2, u3, u4, u5, u6], dtype=float)

    def resid_fn(u):
        return scaling_law_func(X, u) - y

    try:
        sol = least_squares(
            resid_fn,
            init,
            method='trf',
            xtol=1e-8,
            ftol=1e-8,
            max_nfev=10000
        )
        u_opt = sol.x if sol.success else init
    except Exception:
        u_opt = init

    return u_opt
# EVOLVE-BLOCK-END