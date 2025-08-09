import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predicts LM loss from hyperparameters via a generalized power‐law model
    with a small quadratic term in log(lr):
        log(loss) ≈ intercept
                   + w1 * log(lr)
                   + w2 * (log(lr))^2
                   + w3 * log(bsz)
                   + w4 * log(data_size)
                   + w5 * log(non_embedding_param_size)

    Inputs:
      data_points: array of shape (N,4) = [lr, bsz, data_size, param_size]
      params:       array of shape (6,) = [intercept, w1, w2, w3, w4, w5]

    Returns:
      preds: array of shape (N,) of predicted LM losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    # avoid zero or negative before log
    eps = 1e-12
    X = np.maximum(X, eps)
    lr, bsz, data_size, param_size = X.T

    log_lr    = np.log(lr)
    log_bsz   = np.log(bsz)
    log_data  = np.log(data_size)
    log_param = np.log(param_size)

    # build the design matrix
    # features: [1, log(lr), (log(lr))^2, log(bsz), log(data_size), log(param)]
    phi = np.vstack([
        np.ones_like(log_lr),
        log_lr,
        log_lr**2,
        log_bsz,
        log_data,
        log_param
    ]).T  # shape (N,6)

    log_pred = phi.dot(params)    # shape (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the generalized model by ridge‐regularized linear regression in log‐space.

    Inputs:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)

    Returns:
      params: array of shape (6,) = [intercept, w1, w2, w3, w4, w5]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)

    # clip to avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    lr, bsz, data_size, param_size = X.T

    log_lr    = np.log(lr)
    log_bsz   = np.log(bsz)
    log_data  = np.log(data_size)
    log_param = np.log(param_size)

    # build design matrix phi: columns = [1, log_lr, log_lr^2, log_bsz, log_data, log_param]
    N = X.shape[0]
    phi = np.vstack([
        np.ones(N),
        log_lr,
        log_lr**2,
        log_bsz,
        log_data,
        log_param
    ]).T  # shape (N,6)

    # ridge regularization (no penalty on intercept)
    M = phi.shape[1]
    lambda_reg = 1e-6
    I = np.eye(M, dtype=np.float64)
    I[0,0] = 0.0

    A = phi.T.dot(phi) + lambda_reg * I  # (6,6)
    b = phi.T.dot(np.log(y))             # (6,)

    params = np.linalg.solve(A, b)
    return params

# EVOLVE-BLOCK-END