# EVOLVE-BLOCK-START
"""
Refined scaling‐law for LLM training:
We model log(loss) as a hybrid polynomial in the log of each feature,
including first‐order terms for lr and bsz, and both linear and quadratic
terms plus their interaction for data_size (D) and non_embedding_param_size (P).
This keeps the model compact (8 params) and regularizes heavily for better
cross‐configuration generalization.
"""
import numpy as np

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via a tailored log‐domain model.

    Args:
      data_points: array of shape (N,4) with columns
                   [lr, bsz, data_size (D), non_embedding_param_size (P)]
      params:      1D array of length 8:
                   [β0,
                    β_lr, β_bsz, β_D, β_P,
                    β_D2, β_P2, β_DP]
    Returns:
      preds: array of shape (N,) of predicted loss values.
    """
    X = np.asarray(data_points, dtype=float)
    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    # split features
    log_lr  = np.log(X[:, 0])
    log_bsz = np.log(X[:, 1])
    log_D   = np.log(X[:, 2])
    log_P   = np.log(X[:, 3])

    p = np.asarray(params, dtype=float).ravel()
    if p.size != 8:
        raise ValueError(f"Expected 8 parameters, got {p.size}")

    # build compact design matrix Φ
    # columns: [1, log_lr, log_bsz, log_D, log_P, (log_D)^2, (log_P)^2, log_D*log_P]
    N = X.shape[0]
    Phi = np.empty((N, 8), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = log_lr
    Phi[:, 2] = log_bsz
    Phi[:, 3] = log_D
    Phi[:, 4] = log_P
    Phi[:, 5] = log_D * log_D
    Phi[:, 6] = log_P * log_P
    Phi[:, 7] = log_D * log_P

    # linear model in log‐domain
    log_pred = Phi.dot(p)
    # back to original loss scale
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the tailored log‐domain scaling law via ridge‐regularized least squares.

    Args:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params: 1D array of learned parameters of length 8
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    y = np.maximum(y, 1e-12)

    # log‐transform inputs and outputs
    log_lr  = np.log(X[:, 0])
    log_bsz = np.log(X[:, 1])
    log_D   = np.log(X[:, 2])
    log_P   = np.log(X[:, 3])
    log_y   = np.log(y)

    N = X.shape[0]
    # construct design matrix Φ (N×8)
    Phi = np.empty((N, 8), dtype=float)
    Phi[:, 0] = 1.0
    Phi[:, 1] = log_lr
    Phi[:, 2] = log_bsz
    Phi[:, 3] = log_D
    Phi[:, 4] = log_P
    Phi[:, 5] = log_D * log_D
    Phi[:, 6] = log_P * log_P
    Phi[:, 7] = log_D * log_P

    # ridge regularization for stability (penalize all but intercept)
    ridge = 1e-6
    A = Phi.T.dot(Phi)
    # add ridge to diagonal entries except intercept index 0
    diag_idx = np.arange(8)
    A[diag_idx, diag_idx] += ridge
    A[0, 0] -= ridge

    b = Phi.T.dot(log_y)
    # solve for parameters
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END