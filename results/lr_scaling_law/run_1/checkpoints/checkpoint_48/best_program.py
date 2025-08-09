import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters via an extended multiplicative
    power-law model with curvature and interaction terms:

      log(loss) ≈ intercept
                   + w_lr * log(lr)
                   + w_bsz * log(bsz)
                   + w_data * log(data_size)
                   + w_param * log(param_size)
                   + w_lr_bsz * [log(lr)*log(bsz)]
                   + w_data_param * [log(data_size)*log(param_size)]
                   + w_data2 * [log(data_size)]^2
                   + w_param2 * [log(param_size)]^2

    Features:
      x0 = 1
      x1 = log(lr)
      x2 = log(bsz)
      x3 = log(data_size)
      x4 = log(param_size)
      x5 = log(lr)*log(bsz)
      x6 = log(data_size)*log(param_size)
      x7 = (log(data_size))^2
      x8 = (log(param_size))^2

    params: array of shape (9,)
      [intercept,
       w_lr, w_bsz, w_data, w_param,
       w_lr_bsz, w_data_param,
       w_data2, w_param2]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    # avoid non-positive
    eps = 1e-12
    X = np.maximum(X, eps)
    # logs
    log_lr    = np.log(X[:, 0])
    log_bsz   = np.log(X[:, 1])
    log_data  = np.log(X[:, 2])
    log_param = np.log(X[:, 3])

    # interactions & curvature
    inter_lr_bsz     = log_lr * log_bsz
    inter_data_param = log_data * log_param
    sq_data          = log_data * log_data
    sq_param         = log_param * log_param

    # unpack params
    (intercept,
     w_lr, w_bsz, w_data, w_param,
     w_lr_bsz, w_data_param,
     w_data2, w_param2) = params

    # linear predictor in log-space
    log_pred = (
        intercept
        + w_lr          * log_lr
        + w_bsz         * log_bsz
        + w_data        * log_data
        + w_param       * log_param
        + w_lr_bsz      * inter_lr_bsz
        + w_data_param  * inter_data_param
        + w_data2       * sq_data
        + w_param2      * sq_param
    )

    # numerical stability
    log_pred = np.clip(log_pred, -50.0, 50.0)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the extended power-law model in log-space via ridge + Huber-IRLS.

    Model:
      log(loss) ≈ Z @ params

    where Z has columns:
      [1,
       log(lr),
       log(bsz),
       log(data_size),
       log(param_size),
       log(lr)*log(bsz),
       log(data_size)*log(param_size),
       (log(data_size))^2,
       (log(param_size))^2]

    Steps:
      1) Build Z, logy.
      2) Closed-form ridge solve to initialize.
      3) 5 iterations of Huber-weighted IRLS in log-space.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)

    # avoid non-positive
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    # compute log-features
    log_lr    = np.log(X[:, 0])
    log_bsz   = np.log(X[:, 1])
    log_data  = np.log(X[:, 2])
    log_param = np.log(X[:, 3])

    inter_lr_bsz     = log_lr * log_bsz
    inter_data_param = log_data * log_param
    sq_data          = log_data * log_data
    sq_param         = log_param * log_param

    # stack design matrix Z: shape (N,9)
    N = X.shape[0]
    Z = np.column_stack([
        np.ones(N, dtype=np.float64),
        log_lr,
        log_bsz,
        log_data,
        log_param,
        inter_lr_bsz,
        inter_data_param,
        sq_data,
        sq_param
    ])

    # target in log-space
    logy = np.log(y)

    # ridge regularization (no penalty on intercept)
    lambda_reg = 1e-3
    P = Z.shape[1]
    I = np.eye(P, dtype=np.float64)
    I[0, 0] = 0.0

    # initial closed-form solve: (Z^T Z + λI) p = Z^T logy
    A = Z.T.dot(Z) + lambda_reg * I
    b = Z.T.dot(logy)
    try:
        params = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback
        params, *_ = np.linalg.lstsq(Z, logy, rcond=None)

    # Huber-IRLS in log-space
    c = 1.0  # Huber threshold
    for _ in range(5):
        r = Z.dot(params) - logy
        abs_r = np.abs(r)
        # Huber weights: 1 for |r|<=c, c/|r| else
        w = np.where(abs_r <= c, 1.0, c / abs_r)
        sqrt_w = np.sqrt(w)
        Zw = Z * sqrt_w[:, None]
        yw = logy * sqrt_w
        A_w = Zw.T.dot(Zw) + lambda_reg * I
        b_w = Zw.T.dot(yw)
        try:
            params = np.linalg.solve(A_w, b_w)
        except np.linalg.LinAlgError:
            params, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)

    return params

# EVOLVE-BLOCK-END