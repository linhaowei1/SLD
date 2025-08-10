import numpy as np

# EVOLVE-BLOCK-START
"""
Compact 8-parameter log-domain scaling law:

We model log(loss) as a simple linear function of the logs of each feature,
with squared terms for data_size and param_size and their interaction:
    log y = β0
          + β_lr    * log(lr)
          + β_bsz   * log(bsz)
          + β_D     * log(data_size)
          + β_P     * log(param_size)
          + β_D2    * [log(data_size)]^2
          + β_P2    * [log(param_size)]^2
          + β_DP    * log(data_size) * log(param_size)

This 8-parameter form is compact, stable, and fits in closed-form with
a tiny ridge penalty for numerical robustness.
"""

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters via the 8-parameter log-domain model.

    Args:
      data_points: array of shape (N,4) with columns
                   [lr, bsz, data_size, non_embedding_param_size]
      params:      array-like of length 8:
                   [β0, β_lr, β_bsz, β_D, β_P, β_D2, β_P2, β_DP]
    Returns:
      preds: array of shape (N,) of predicted LM loss values.
    """
    X = np.asarray(data_points, dtype=float)
    # allow single point input
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 4:
        raise ValueError(f"Expected data_points shape (N,4), got {X.shape}")

    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    lr, bsz, D, Psize = X[:,0], X[:,1], X[:,2], X[:,3]

    # log-transform
    log_lr  = np.log(lr)
    log_bsz = np.log(bsz)
    log_D   = np.log(D)
    log_P   = np.log(Psize)

    # build design matrix Φ of shape (N,8)
    # columns = [1, log_lr, log_bsz, log_D, log_P, log_D^2, log_P^2, log_D*log_P]
    Phi = np.column_stack([
        np.ones_like(log_lr),
        log_lr,
        log_bsz,
        log_D,
        log_P,
        log_D**2,
        log_P**2,
        log_D * log_P
    ])

    p = np.asarray(params, dtype=float).ravel()
    if p.size != 8:
        raise ValueError(f"Expected 8 parameters, got {p.size}")

    # linear predictor in log-loss domain
    log_pred = Phi.dot(p)
    # back to loss domain
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 8-parameter log-domain scaling law via ridge-regularized
    least squares in the log-loss domain.

    Args:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params: 1D array of length 8 (β0, β_lr, β_bsz, β_D, β_P, β_D2, β_P2, β_DP)
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()

    # allow single point input
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of data points and loss values must match.")
    if X.shape[1] != 4:
        raise ValueError(f"Expected data_points shape (N,4), got {X.shape}")

    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    y = np.maximum(y, 1e-12)

    lr, bsz, D, Psize = X[:,0], X[:,1], X[:,2], X[:,3]
    log_lr  = np.log(lr)
    log_bsz = np.log(bsz)
    log_D   = np.log(D)
    log_P   = np.log(Psize)
    log_y   = np.log(y)

    N = X.shape[0]
    # assemble design matrix Φ
    Phi = np.column_stack([
        np.ones(N),
        log_lr,
        log_bsz,
        log_D,
        log_P,
        log_D**2,
        log_P**2,
        log_D * log_P
    ])

    # closed-form ridge solution: solve (ΦᵀΦ + λ·I') p = Φᵀ log_y
    ridge = 1e-6
    A = Phi.T.dot(Phi)
    # apply small ridge only to non-intercept terms
    A[1:,1:] += ridge
    b = Phi.T.dot(log_y)

    # solve for parameters
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END