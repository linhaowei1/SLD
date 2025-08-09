import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters via an extended power‐law model
    in log‐space with three interaction terms:
        log(loss) ≈ intercept
                   + w_lr * log(lr)
                   + w_bsz * log(bsz)
                   + w_data * log(data_size)
                   + w_param * log(non_embedding_param_size)
                   + w_lr_bsz * [log(lr) * log(bsz)]
                   + w_lr_data * [log(lr) * log(data_size)]
                   + w_data_param * [log(data_size) * log(non_embedding_param_size)]
    Returns loss = exp(log_pred).

    Inputs:
      data_points: array of shape (N,4) columns = [lr, bsz, data_size, param_size]
      params:       array of shape (8,)
                    [intercept,
                     w_lr, w_bsz, w_data, w_param,
                     w_lr_bsz, w_lr_data, w_data_param]
    Returns:
      preds: array of shape (N,) of predicted LM losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    # clip to avoid log(0) or negatives
    eps = 1e-12
    X = np.maximum(X, eps)

    # unpack hyperparameters
    lr         = X[:, 0]
    bsz        = X[:, 1]
    data_size  = X[:, 2]
    param_size = X[:, 3]

    # compute logs
    log_lr    = np.log(lr)
    log_bsz   = np.log(bsz)
    log_data  = np.log(data_size)
    log_param = np.log(param_size)

    # interactions
    lr_bsz    = log_lr * log_bsz
    lr_data   = log_lr * log_data
    data_param= log_data * log_param

    # unpack parameters
    (intercept,
     w_lr, w_bsz, w_data, w_param,
     w_lr_bsz, w_lr_data, w_data_param) = params

    # linear model in log-space
    log_pred = (intercept
                + w_lr       * log_lr
                + w_bsz      * log_bsz
                + w_data     * log_data
                + w_param    * log_param
                + w_lr_bsz   * lr_bsz
                + w_lr_data  * lr_data
                + w_data_param * data_param)

    # back to original scale
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the extended power‐law model by ridge‐regularized linear regression in log‐space.
    Features: [1,
               log(lr), log(bsz), log(data_size), log(param_size),
               log(lr)*log(bsz), log(lr)*log(data_size), log(data_size)*log(param_size)]
    Solves (Φ^T Φ + λI) θ = Φ^T log(loss) for θ.

    Inputs:
      data_points: array of shape (N,4) = [lr, bsz, data_size, param_size]
      loss_values: array of shape (N,)
    Returns:
      params: array of shape (8,) as in scaling_law_func docstring
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)

    # clip to avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    # unpack and log-transform
    lr         = X[:, 0]
    bsz        = X[:, 1]
    data_size  = X[:, 2]
    param_size = X[:, 3]

    log_lr    = np.log(lr)
    log_bsz   = np.log(bsz)
    log_data  = np.log(data_size)
    log_param = np.log(param_size)

    # build interaction terms
    lr_bsz     = log_lr * log_bsz
    lr_data    = log_lr * log_data
    data_param = log_data * log_param

    # design matrix Φ shape (N,8)
    N = X.shape[0]
    ones = np.ones(N, dtype=np.float64)
    phi = np.column_stack([
        ones,
        log_lr,
        log_bsz,
        log_data,
        log_param,
        lr_bsz,
        lr_data,
        data_param
    ])

    # target in log-space
    log_y = np.log(y)

    # ridge regularization (no penalty on intercept)
    M = phi.shape[1]
    lambda_reg = 1e-3
    I = np.eye(M, dtype=np.float64)
    I[0, 0] = 0.0  # do not regularize intercept

    # solve normal equations
    A = phi.T.dot(phi) + lambda_reg * I
    b = phi.T.dot(log_y)
    params = np.linalg.solve(A, b)

    return params

# EVOLVE-BLOCK-END