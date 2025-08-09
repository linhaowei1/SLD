import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
"""
Enhanced scaling law: additive floor + extended log‐space regression with
quadratic lr term and data‐param interaction.

Model:
  loss_pred = loss_floor + exp(
      intercept
      + w_lr1 * log(lr)
      + w_lr2 * (log(lr))^2
      + w_bsz * log(bsz)
      + w_data * log(data_size)
      + w_param * log(non_embedding_param_size)
      + w_cross * log(data_size) * log(non_embedding_param_size)
  )

Fitting:
  1) Ridge‐regularized linear regression in augmented log‐space
  2) Robust Huber least‐squares refinement on original loss‐space
"""
def scaling_law_func(data_points, params):
    """
    data_points: (N,4) = [lr, bsz, data_size, non_embedding_param_size]
    params: (8,) = [
      loss_floor,
      intercept,
      w_lr1, w_lr2,
      w_bsz,
      w_data,
      w_param,
      w_cross
    ]
    returns: (N,) predicted loss
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    # clamp to avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)

    # unpack
    loss_floor = params[0]
    intercept  = params[1]
    w_lr1, w_lr2, w_bsz, w_data, w_param, w_cross = params[2:]

    # compute logs
    log_lr    = np.log(X[:, 0])
    log_bsz   = np.log(X[:, 1])
    log_data  = np.log(X[:, 2])
    log_param = np.log(X[:, 3])

    # quadratic and interaction features
    lr_quad    = log_lr * log_lr
    data_param = log_data * log_param

    lin_term = (
        intercept
        + w_lr1 * log_lr
        + w_lr2 * lr_quad
        + w_bsz * log_bsz
        + w_data * log_data
        + w_param * log_param
        + w_cross * data_param
    )
    return loss_floor + np.exp(lin_term)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the extended scaling law.
    Returns params of shape (8,).
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)
    N, F = X.shape
    assert F == 4, "Expect 4 features per point"

    # clamp
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    # build augmented log‐space design matrix
    log_lr    = np.log(X[:, 0])
    log_bsz   = np.log(X[:, 1])
    log_data  = np.log(X[:, 2])
    log_param = np.log(X[:, 3])

    lr_quad    = log_lr * log_lr
    data_param = log_data * log_param

    # columns: [1, log_lr, lr_quad, log_bsz, log_data, log_param, data_param]
    ones = np.ones((N, 1), dtype=np.float64)
    design = np.hstack([
        ones,
        log_lr.reshape(-1, 1),
        lr_quad.reshape(-1, 1),
        log_bsz.reshape(-1, 1),
        log_data.reshape(-1, 1),
        log_param.reshape(-1, 1),
        data_param.reshape(-1, 1)
    ])  # shape (N,7)

    # initial linear solve in log‐loss space
    logy = np.log(y)
    # ridge regularization (no penalty on intercept)
    D = design.shape[1]
    lambda_reg = 1e-6
    I = np.eye(D, dtype=np.float64)
    I[0, 0] = 0.0
    A = design.T.dot(design) + lambda_reg * I
    b = design.T.dot(logy)
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    intercept0 = sol[0]
    w_init     = sol[1:]  # length 6

    # initial floor: half min(y)
    loss_floor0 = max(0.0, 0.5 * np.min(y))

    # pack initial params: [floor, intercept, w_lr1, w_lr2, w_bsz, w_data, w_param, w_cross]
    p0 = np.concatenate(([loss_floor0, intercept0], w_init))

    # bounds: loss_floor ∈ [0, min(y)], others unconstrained
    lower = np.concatenate(([0.0, -np.inf], [-np.inf] * (D - 1)))
    upper = np.concatenate(([np.min(y), np.inf], [np.inf] * (D - 1)))

    # residual function in original space
    def residuals(p):
        return scaling_law_func(X, p) - y

    # robust refinement
    try:
        res = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            loss='huber',
            f_scale=0.5,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=5000
        )
        p_opt = res.x if res.success else p0
    except Exception:
        p_opt = p0

    return p_opt
# EVOLVE-BLOCK-END