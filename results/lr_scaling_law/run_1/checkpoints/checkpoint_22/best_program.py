import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predicts LM loss from hyperparameters via an extended multiplicative 
    power‐law model with two key interaction terms:
        log(loss) ≈ intercept 
                     + w_lr * log(lr) 
                     + w_bsz * log(bsz)
                     + w_data * log(data_size)
                     + w_param * log(non_embedding_param_size)
                     + w_lr_bsz * [log(lr) * log(bsz)]
                     + w_data_param * [log(data_size) * log(non_embedding_param_size)]
    Returns loss = exp(log_pred).
    Inputs:
      data_points: array of shape (N,4)
      params:       array of shape (7,)
                    [intercept,
                     w_lr, w_bsz, w_data, w_param,
                     w_lr_bsz, w_data_param]
    Returns:
      preds: array of shape (N,) of predicted LM losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    # avoid zero or negative input to log
    eps = 1e-12
    X = np.maximum(X, eps)
    # extract columns
    lr = X[:, 0]
    bsz = X[:, 1]
    data_size = X[:, 2]
    param_size = X[:, 3]

    # compute logs
    log_lr = np.log(lr)
    log_bsz = np.log(bsz)
    log_data = np.log(data_size)
    log_param = np.log(param_size)

    # interactions
    inter_lr_bsz = log_lr * log_bsz
    inter_data_param = log_data * log_param

    # unpack params
    intercept     = params[0]
    w_lr          = params[1]
    w_bsz         = params[2]
    w_data        = params[3]
    w_param       = params[4]
    w_lr_bsz      = params[5]
    w_data_param  = params[6]

    # linear combination in log-space
    log_pred = (intercept
                + w_lr * log_lr
                + w_bsz * log_bsz
                + w_data * log_data
                + w_param * log_param
                + w_lr_bsz * inter_lr_bsz
                + w_data_param * inter_data_param)

    # back to original scale
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fits the extended power‐law model by ridge-regression in log-space 
    with two interaction terms.
    Builds a design matrix:
      [1,
       log(lr), log(bsz), log(data_size), log(param_size),
       log(lr)*log(bsz), log(data_size)*log(param_size)]
    and solves (X^T X + λI) p = X^T y for p in log-loss space.
    Inputs:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params: array of shape (7,) = 
              [intercept, w_lr, w_bsz, w_data, w_param, w_lr_bsz, w_data_param]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)

    # avoid zeros
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    # logs
    log_lr    = np.log(X[:, 0])
    log_bsz   = np.log(X[:, 1])
    log_data  = np.log(X[:, 2])
    log_param = np.log(X[:, 3])

    # interactions
    inter_lr_bsz     = log_lr * log_bsz
    inter_data_param = log_data * log_param

    # build design matrix: shape (N,7)
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    design = np.column_stack([
        ones,
        log_lr[:, None],
        log_bsz[:, None],
        log_data[:, None],
        log_param[:, None],
        inter_lr_bsz[:, None],
        inter_data_param[:, None]
    ])

    # target in log-space
    logy = np.log(y)

    # ridge regularization (do not penalize intercept)
    dim = design.shape[1]
    lambda_reg = 1e-3
    I = np.eye(dim, dtype=np.float64)
    I[0, 0] = 0.0

    # normal equations
    A = design.T.dot(design) + lambda_reg * I
    b = design.T.dot(logy)

    # solve for parameters
    params = np.linalg.solve(A, b)

    return params

# EVOLVE-BLOCK-END