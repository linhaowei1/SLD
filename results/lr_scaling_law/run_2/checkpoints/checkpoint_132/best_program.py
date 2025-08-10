# EVOLVE-BLOCK-START
"""
Refined scaling‐law model for LLM training hyperparameters:
We predict LM loss by fitting log(loss) as a normalized quadratic polynomial
in the log‐domain of each feature (learning rate, batch size, data size,
non‐embedding parameter size).  We normalize each log‐feature to [0,1]
based on known domain ranges for numerical stability, build a design matrix
with intercept, linear, squared, and pairwise interaction terms, and solve
the normal equations with adaptive ridge regularization.
"""
import numpy as np

# Precomputed log‐domain minima and maxima for each feature:
#   [learning_rate, batch_size, data_size, non_embedding_param_size]
_LOG_F_MINS = np.log(np.array([
    1.2e-4,   # lr min
    16.0,     # bsz min
    4e9,      # data_size min
    2.14e8    # non_embedding_param_size min
], dtype=float))
_LOG_F_MAXS = np.log(np.array([
    2.2e-2,   # lr max
    4096.0,   # bsz max
    1e11,     # data_size max
    1e9       # non_embedding_param_size max
], dtype=float))
_LOG_F_RANGES = _LOG_F_MAXS - _LOG_F_MINS  # used to normalize to [0,1]

def _build_design_matrix(logX_norm):
    """
    Build a design matrix Φ for a second‐order polynomial in the normalized
    log‐domain.  Columns:
      [1,
       logX_norm_i for each feature (4),
       (logX_norm_i)^2 for each feature (4),
       logX_norm_i * logX_norm_j for all i<j (6)
      ]
    """
    N, F = logX_norm.shape
    # total params: intercept + F linear + F squared + F*(F-1)/2 interactions
    P = 1 + F + F + (F * (F - 1)) // 2
    Phi = np.empty((N, P), dtype=logX_norm.dtype)
    # intercept
    Phi[:, 0] = 1.0
    # linear terms
    Phi[:, 1:1+F] = logX_norm
    # squared terms
    start_sq = 1 + F
    Phi[:, start_sq:start_sq+F] = logX_norm**2
    # pairwise interactions
    idx = start_sq + F
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, idx] = logX_norm[:, i] * logX_norm[:, j]
            idx += 1
    return Phi

def scaling_law_func(data_points, params):
    """
    Predict language‐model loss from hyperparameters via the learned
    normalized‐quadratic scaling law.

    Args:
      data_points: array of shape (N,4) with columns
                   [lr, bsz, data_size, non_embedding_param_size]
      params:      array of length P = 1 + 2*4 + 6 = 15

    Returns:
      preds: array of shape (N,) of predicted LM loss values
    """
    X = np.asarray(data_points, dtype=float)
    # support single-row input
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 4:
        raise ValueError(f"Expected input with 4 features, got shape {X.shape}")
    # floor to avoid log(0)
    X = np.clip(X, 1e-12, None)
    # log‐transform and normalize each feature to [0,1]
    logX = np.log(X)
    logX_norm = (logX - _LOG_F_MINS) / _LOG_F_RANGES
    # clip to [0,1] to guard against slight out‐of‐range values
    logX_norm = np.clip(logX_norm, 0.0, 1.0)
    # build design matrix and predict
    Phi = _build_design_matrix(logX_norm)       # shape (N,15)
    p = np.asarray(params, dtype=float).ravel()
    if p.shape[0] != Phi.shape[1]:
        raise ValueError(f"Expected {Phi.shape[1]} params, got {p.shape[0]}")
    log_pred = Phi.dot(p)                       # shape (N,)
    return np.exp(log_pred)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the normalized‐quadratic scaling law via closed‐form ridge regression.

    Args:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)

    Returns:
      params: array of length 15 of learned coefficients in log‐loss domain
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()
    # support single-row input
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of data points and loss values must match.")
    # floor to avoid log(0)
    X = np.clip(X, 1e-12, None)
    y = np.clip(y, 1e-12, None)
    # log‐transform and normalize features
    logX = np.log(X)
    logX_norm = (logX - _LOG_F_MINS) / _LOG_F_RANGES
    logX_norm = np.clip(logX_norm, 0.0, 1.0)
    # build design matrix
    Phi = _build_design_matrix(logX_norm)       # shape (N,15)
    # transform target to log‐domain
    logy = np.log(y)
    # normal equations
    A = Phi.T.dot(Phi)                          # shape (15,15)
    b = Phi.T.dot(logy)                         # shape (15,)
    # adaptive ridge: scale by trace(A)/P for balanced regularization
    P = A.shape[0]
    ridge = 1e-6 * np.trace(A) / P
    # add ridge to diagonal except intercept
    diag_idx = np.diag_indices(P)
    A[diag_idx] += ridge
    A[0, 0] -= ridge  # no penalty on intercept
    # solve for parameters
    params = np.linalg.solve(A, b)
    return params
# EVOLVE-BLOCK-END