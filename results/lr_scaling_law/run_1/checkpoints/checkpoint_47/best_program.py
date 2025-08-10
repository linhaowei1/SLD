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

# mid‐point and half‐range for each log‐feature
_LOG_MEAN  = 0.5 * (_LOG_MIN + _LOG_MAX)
_LOG_SCALE = 0.5 * (_LOG_MAX - _LOG_MIN)

def _build_design_matrix(X):
    """
    Build a normalized‐log polynomial design matrix with:
      - intercept
      - linear terms z_i
      - quadratic terms z_i^2
      - pairwise interactions z_i * z_j for i<j

    X: (N,4) array of raw features [lr, bsz, data_size, non_embed_param_size]
    returns: A (N,15) design matrix
    """
    eps = 1e-12
    # compute and normalize logs
    logs = np.log(X + eps)                         # (N,4)
    z = (logs - _LOG_MEAN) / _LOG_SCALE            # normalize to ~[-1,1]
    N, F = z.shape

    # intercept
    cols = [np.ones((N,1), dtype=np.float64)]
    # linear terms
    cols.append(z)
    # quadratic terms
    cols.append(z**2)
    # pairwise interactions
    inters = []
    for i in range(F):
        for j in range(i+1, F):
            inters.append((z[:, i] * z[:, j])[:, None])
    if inters:
        cols.append(np.hstack(inters))

    return np.hstack(cols)  # shape (N, 1 + 4 + 4 + 6 = 15)


def scaling_law_func(data_points, params):
    """
    Predict language‐model loss via a 2nd‐degree polynomial
    in normalized log‐features with interactions:
      log_loss = A θ
      loss     = exp(log_loss)
    Inputs:
      data_points: array‐like of shape (N,4)
      params:      array‐like of shape (15,)
    Returns:
      preds: (N,) predicted LM losses
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 4:
        raise ValueError(f"scaling_law_func: expected shape (N,4), got {X.shape}")

    theta = np.asarray(params, dtype=np.float64).ravel()
    A = _build_design_matrix(X)           # (N,15)
    if theta.size != A.shape[1]:
        raise ValueError(f"scaling_law_func: expected {A.shape[1]} params, got {theta.size}")

    log_pred = A.dot(theta)               # (N,)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 15‐parameter polynomial scaling law by solving
    a ridge‐regularized least squares problem in log(loss):
      minimize ||A θ − log(y)||^2 + λ ||θ_{1:}||^2
    (no regularization on the intercept θ0).
    Inputs:
      data_points: (N,4) array
      loss_values: (N,)   array
    Returns:
      theta_opt: (15,) optimized parameters
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 4:
        raise ValueError(f"fit_scaling_law: expected shape (N,4), got {X.shape}")

    y = np.asarray(loss_values, dtype=np.float64).ravel()
    if X.shape[0] != y.size:
        raise ValueError("fit_scaling_law: number of data points and losses must match")

    # build design matrix and target
    A = _build_design_matrix(X)            # (N,15)
    eps = 1e-12
    y_log = np.log(y + eps)                # (N,)

    # normal equations with adaptive ridge regularization
    ATA = A.T.dot(A)                       # (15,15)
    P = ATA.shape[0]
    # scale lambda by average diagonal to adapt to data scale
    lam = 1e-4 * np.trace(ATA) / P
    reg = lam * np.eye(P, dtype=np.float64)
    reg[0,0] = 0.0                         # no penalty on intercept

    ATA_reg = ATA + reg
    ATy = A.T.dot(y_log)

    # solve robustly
    try:
        theta_opt = np.linalg.solve(ATA_reg, ATy)
    except np.linalg.LinAlgError:
        theta_opt = np.linalg.pinv(ATA_reg).dot(ATy)

    return theta_opt
# EVOLVE-BLOCK-END