import numpy as np

def scaling_law_func(data_points, params):
    """
    Predicts LM loss from hyperparameters via a multiplicative power‐law model:
        loss ≈ exp(intercept + Σ_i w_i * log(x_i))
    where x = [lr, bsz, data_size, non_embedding_param_size].

    Inputs:
      data_points: array of shape (N,4)
      params:       array of shape (5,) = [intercept, w_lr, w_bsz, w_data, w_param]

    Returns:
      preds: array of shape (N,) of predicted LM losses
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    # clip to avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)
    logX = np.log(X)                                             # (N,4)

    intercept = float(params[0])                                 # scalar
    weights   = np.asarray(params[1:], dtype=np.float64)         # (4,)

    # linear model in log-space
    log_pred = intercept + logX.dot(weights)                     # (N,)

    # numerical stability: clip log_pred to avoid overflow/underflow
    log_pred = np.clip(log_pred, -50.0, 50.0)

    # map back to original scale
    preds = np.exp(log_pred)                                     # (N,)
    return preds


def fit_scaling_law(data_points, loss_values):
    """
    Fits the power‐law model by linear regression in log-space with feature
    normalization and ridge regularization:
        log(loss) ≈ intercept + Σ_i w_i * log(x_i)

    Inputs:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)

    Returns:
      params: array of shape (5,) = [intercept, w_lr, w_bsz, w_data, w_param]
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))  # (N,4)
    y = np.asarray(loss_values, dtype=np.float64)                 # (N,)

    # clip to avoid log(0)
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    # take logs
    logX = np.log(X)                                              # (N,4)
    logy = np.log(y)                                              # (N,)

    N, F = logX.shape   # F should be 4

    # normalize features for numeric stability
    mu    = logX.mean(axis=0)                                     # (4,)
    sigma = logX.std(axis=0)                                      # (4,)
    sigma[sigma < eps] = 1.0

    Z = (logX - mu) / sigma                                       # (N,4)

    # build design matrix [1, z1, z2, z3, z4]
    ones = np.ones((N, 1), dtype=np.float64)
    D = np.hstack([ones, Z])                                      # (N,5)

    # ridge regularization (do not penalize intercept)
    lam = 1e-3
    P = F + 1
    I = np.eye(P, dtype=np.float64)
    I[0, 0] = 0.0

    # normal equations: (D^T D + λI) p_z = D^T logy
    A = D.T.dot(D) + lam * I                                      # (5,5)
    b = D.T.dot(logy)                                             # (5,)

    # solve for normalized-parameter vector p_z = [p0, p1, ..., p4]
    p_z = np.linalg.solve(A, b)                                   # (5,)

    # convert back to original weights: w_i = p_z[i+1] / sigma[i]
    weights = p_z[1:] / sigma                                     # (4,)
    # intercept adjustment: p_z[0] - Σ_i (w_i * mu_i)
    intercept = p_z[0] - np.dot(weights, mu)

    # pack params for scaling_law_func
    params = np.empty(F + 1, dtype=np.float64)
    params[0]  = intercept
    params[1:] = weights

    return params