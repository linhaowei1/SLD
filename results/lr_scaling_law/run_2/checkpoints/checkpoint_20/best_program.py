import numpy as np

# We work in log‐domain of each feature, then center to zero mean and scale to unit half‐range,
# so logX_norm ≈ in [−1,1].  This reduces multicollinearity and yields numerically stable polynomials.

# Precomputed log‐domain minima and maxima for each feature:
#   [learning_rate, batch_size, data_size, non_embedding_param_size]
_LOG_F_MIN = np.log(np.array([1.2e-4, 16.0, 4e9, 2.14e8], dtype=float))
_LOG_F_MAX = np.log(np.array([2.2e-2, 4096.0, 1e11, 1e9], dtype=float))

# Compute midpoints and half‐ranges in log‐domain
_LOG_F_MID   = (_LOG_F_MIN + _LOG_F_MAX) * 0.5
_LOG_F_HALF  = (_LOG_F_MAX - _LOG_F_MIN) * 0.5

def _build_design_matrix(logX_norm):
    """
    Build design matrix for a 2nd‐order polynomial in the F=4 normalized log‐features:
      - intercept
      - F linear terms
      - F squared terms
      - F*(F-1)/2 pairwise products
    Returns Phi of shape (N, 1 + F + F + F*(F-1)//2).
    """
    N, F = logX_norm.shape
    P = 1 + F + F + (F*(F-1))//2
    Phi = np.empty((N, P), dtype=float)
    col = 0

    # intercept
    Phi[:, col] = 1.0
    col += 1

    # linear terms
    Phi[:, col:col+F] = logX_norm
    col += F

    # squared terms
    Phi[:, col:col+F] = logX_norm**2
    col += F

    # pairwise interaction terms
    for i in range(F):
        for j in range(i+1, F):
            Phi[:, col] = logX_norm[:, i] * logX_norm[:, j]
            col += 1

    return Phi

def scaling_law_func(data_points, params):
    """
    Predict LM loss from hyperparameters via a zero‐centered, scaled,
    2nd‐order polynomial in log‐domain.
    
    Inputs:
      data_points: array of shape (N,4) with columns
                   [lr, bsz, data_size, non_embedding_param_size]
      params:      1D array of length P = 1 + 4 + 4 + 6 = 15
    Returns:
      preds:       array of shape (N,) of predicted losses
    """
    X = np.array(data_points, dtype=float)
    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    logX = np.log(X)  # (N,4)

    # normalize to zero mean, half‐range = 1
    logX_norm = (logX - _LOG_F_MID) / _LOG_F_HALF

    # build polynomial design matrix
    Phi = _build_design_matrix(logX_norm)

    p = np.asarray(params, dtype=float).ravel()
    if p.shape[0] != Phi.shape[1]:
        raise ValueError(f"Expected {Phi.shape[1]} params but got {p.shape[0]}")

    # linear model in log‐loss domain
    log_pred = Phi.dot(p)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the above polynomial scaling law via ridge‐regularized normal equations.
    
    Inputs:
      data_points: array of shape (N,4)
      loss_values: array of shape (N,)
    Returns:
      params:      1D array of length 15
    """
    X = np.array(data_points, dtype=float)
    y = np.array(loss_values, dtype=float)

    # floor to avoid log(0)
    X = np.maximum(X, 1e-12)
    y = np.maximum(y, 1e-12)

    logX = np.log(X)    # (N,4)
    logy = np.log(y)    # (N,)

    # normalize features
    logX_norm = (logX - _LOG_F_MID) / _LOG_F_HALF

    # build design
    Phi = _build_design_matrix(logX_norm)  # (N,15)
    N, P = Phi.shape

    # form normal equations
    A = Phi.T.dot(Phi)
    b = Phi.T.dot(logy)

    # adaptive ridge: scale by average diag magnitude
    avg_diag = np.trace(A) / P
    ridge = 1e-3 * avg_diag

    # apply ridge to all but intercept
    diag_idx = np.diag_indices(P)
    A[diag_idx] += ridge
    A[0,0] -= ridge  # no penalty on intercept

    # solve for parameters
    params = np.linalg.solve(A, b)
    return params