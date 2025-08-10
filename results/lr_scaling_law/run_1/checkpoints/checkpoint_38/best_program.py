import numpy as np

# EVOLVE-BLOCK-START
# Pre‐defined log‐ranges for normalization (from problem description)
_LOG_MIN = np.log(np.array([
    1.2e-4,   # lr
    16.0,     # bsz
    4e9,      # data_size
    2.14e8    # non_embedding_param_size
], dtype=np.float64))

_LOG_MAX = np.log(np.array([
    2.2e-2,   # lr
    4096.0,   # bsz
    1e11,     # data_size
    1e9       # non_embedding_param_size
], dtype=np.float64))

# Compute normalization constants
_LOG_MEAN  = 0.5 * (_LOG_MIN + _LOG_MAX)
_LOG_SCALE = 0.5 * (_LOG_MAX - _LOG_MIN)

def _build_design_matrix(X):
    """
    Build a polynomial design matrix on normalized log‐features:
      z = (log(X) - mean) / scale
    Features: [1,
               z1, z2, z3, z4,
               z1^2, z2^2, z3^2, z4^2,
               z1*z2, z1*z3, z1*z4,
               z2*z3, z2*z4, z3*z4]
    Returns A with shape (N,15).
    """
    eps = 1e-12
    logs = np.log(X + eps)                    # (N,4)
    z = (logs - _LOG_MEAN) / _LOG_SCALE        # normalize to ~[-1,1]
    N = X.shape[0]
    # intercept
    feats = [np.ones(N, dtype=np.float64)]
    # linear terms
    feats += [z[:, i] for i in range(4)]
    # quadratic terms
    feats += [z[:, i] * z[:, i] for i in range(4)]
    # pairwise interactions
    for i in range(4):
        for j in range(i + 1, 4):
            feats.append(z[:, i] * z[:, j])
    return np.column_stack(feats)             # (N,15)


def scaling_law_func(data_points, params):
    """
    Predict language‐model loss via a 2nd‐degree polynomial in normalized log‐features.
    Inputs:
      data_points: array‐like of shape (N,4) columns [lr, bsz, data_size, non_embed_params]
      params:      array‐like of length 15
    Output:
      preds: (N,) predicted losses
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 4:
        raise ValueError(f"Expected 4 features, got {X.shape[1]}")
    theta = np.asarray(params, dtype=np.float64).ravel()
    if theta.size != 15:
        raise ValueError(f"Expected 15 parameters, got {theta.size}")
    A = _build_design_matrix(X)               # (N,15)
    log_pred = A.dot(theta)                   # (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 15‐parameter polynomial scaling law via ridge‐regularized least squares
    on the log‐transformed loss:
      minimize ||A θ − log(y)||^2 + λ ||θ_{1:}||^2
    (No penalty on the intercept θ0.)
    Inputs:
      data_points: (N,4) array
      loss_values: (N,)   array of observed losses
    Output:
      theta_opt: (15,) optimized parameter vector
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=np.float64).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of data points and losses must match")

    A = _build_design_matrix(X)               # (N,15)
    eps = 1e-12
    y_log = np.log(y + eps)

    # ridge regularization
    P = A.shape[1]                            # 15
    lam = 1e-4                                # regularization strength
    reg = lam * np.eye(P, dtype=np.float64)
    reg[0, 0] = 0.0                          # no penalty on intercept

    # solve (A^T A + reg) θ = A^T y_log
    ATA = A.T.dot(A) + reg
    ATy = A.T.dot(y_log)
    theta_opt = np.linalg.solve(ATA, ATy)
    return theta_opt
# EVOLVE-BLOCK-END