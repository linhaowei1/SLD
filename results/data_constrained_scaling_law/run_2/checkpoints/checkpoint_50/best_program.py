import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict scalar loss from features [tokens, params, unique_tokens]
    using a sum of three decaying power‐law terms plus an asymptotic floor.

    data_points: array‐like of shape (N,3)
    params:      array‐like of shape (7,) or (T,7) containing raw parameters
                 r0: log(floor)
                 r1: log(amp_tokens), r2: log(decay_tokens)
                 r3: log(amp_params), r4: log(decay_params)
                 r5: log(amp_unique), r6: log(decay_unique)
    returns:     predicted loss of shape (N,) or (N,T)
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    # clamp inputs to avoid zero/div
    eps = 1e-12
    tkn = np.maximum(X[:, 0], eps)
    prm = np.maximum(X[:, 1], eps)
    ut  = np.maximum(X[:, 2], eps)

    p = np.asarray(params, dtype=float)
    if p.ndim == 1:
        p = p[None, :]    # (1,7)
    T, P = p.shape
    if P != 7:
        raise ValueError("Expected params of length 7, got shape %s" % (p.shape,))

    # unpack raw parameters
    r0 = p[:, 0]   # floor log
    r1 = p[:, 1]; r2 = p[:, 2]
    r3 = p[:, 3]; r4 = p[:, 4]
    r5 = p[:, 5]; r6 = p[:, 6]

    # transform to positive
    floor = np.exp(r0)              # (T,)
    amp_t = np.exp(r1); decay_t = np.exp(r2)
    amp_p = np.exp(r3); decay_p = np.exp(r4)
    amp_u = np.exp(r5); decay_u = np.exp(r6)

    # compute power‐law decays
    # shape (N,T)
    ct = amp_t[None, :] * tkn[:, None] ** (-decay_t[None, :])
    cp = amp_p[None, :] * prm[:, None] ** (-decay_p[None, :])
    cu = amp_u[None, :] * ut[:, None]  ** (-decay_u[None, :])

    preds = floor[None, :] + ct + cp + cu

    # return (N,) if single param‐vector
    return preds[:, 0] if preds.shape[1] == 1 else preds

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 7 raw parameters [r0..r6] of scaling_law_func by
    minimizing robust squared error.

    data_points: (N,3), loss_values: (N,)
    returns:      array of length 7 (optimized raw parameters)
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 3:
        raise ValueError("data_points must have shape (N,3)")
    y = np.asarray(loss_values, dtype=float).ravel()
    N = X.shape[0]
    if y.shape[0] != N:
        raise ValueError("Mismatched data_points and loss_values lengths")

    # initial raw parameters
    y_min, y_mean = np.min(y), np.mean(y)
    # floor below min to allow slight below
    init_r0 = np.log(max(y_min * 0.9, 1e-8))
    # split residual equally across three amplitudes
    resid = max(y_mean - np.exp(init_r0), 1e-8)
    amp0 = resid / 3.0
    init_r1 = np.log(max(amp0, 1e-8))
    init_r3 = np.log(max(amp0, 1e-8))
    init_r5 = np.log(max(amp0, 1e-8))
    # initial decays moderate (power -0.5)
    init_r2 = np.log(0.5)
    init_r4 = np.log(0.5)
    init_r6 = np.log(0.5)

    init = np.array([init_r0, init_r1, init_r2,
                     init_r3, init_r4, init_r5, init_r6], dtype=float)

    # define residuals function
    def resid_fn(raw):
        pred = scaling_law_func(X, raw)
        return pred - y

    # robust non‐linear least squares
    sol = least_squares(
        resid_fn,
        init,
        method='trf',
        loss='soft_l1',
        f_scale=0.1,
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=2000
    )

    raw_opt = sol.x if sol.success else init
    return raw_opt
# EVOLVE-BLOCK-END