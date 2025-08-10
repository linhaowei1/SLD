import numpy as np

# EVOLVE-BLOCK-START

# Pre‐defined log‐ranges for normalization (from problem description)
_LOG_MIN = np.log(np.array([
    1.2e-4,   # learning rate minimum
    16.0,     # batch size minimum
    4e9,      # data size minimum
    2.14e8    # non‐embedding parameter size minimum
], dtype=np.float64))

_LOG_MAX = np.log(np.array([
    2.2e-2,   # learning rate maximum
    4096.0,   # batch size maximum
    1e11,     # data size maximum
    1e9       # non‐embedding parameter size maximum
], dtype=np.float64))

# Compute midpoint and half‐range for each feature (for normalization to ~[-1,1])
_LOG_MEAN = 0.5 * (_LOG_MIN + _LOG_MAX)
_LOG_SCALE = 0.5 * (_LOG_MAX - _LOG_MIN)

def _build_design_matrix(X):
    """
    Build a numerically stable design matrix using Chebyshev‐basis polynomials
    on normalized log‐features z_i = (log x_i − _LOG_MEAN_i)/_LOG_SCALE_i.
    Features for each sample:
      [1,
       z1, z2, z3, z4,
       T2(z1), T2(z2), T2(z3), T2(z4),
       z1*z2, z1*z3, z1*z4, z2*z3, z2*z4, z3*z4]
    where T2(z) = 2*z^2 − 1 is the 2nd‐degree Chebyshev polynomial.
    Returns:
      A: ndarray of shape (N, 15)
    """
    eps = 1e-12
    # Safe log
    logs = np.log(X + eps)
    # Normalize to roughly [-1,1]
    z = (logs - _LOG_MEAN) / _LOG_SCALE
    N, F = z.shape  # F should be 4

    # Intercept
    cols = [np.ones((N, 1), dtype=np.float64)]
    # Linear Chebyshev feature = z
    cols.append(z)
    # Quadratic Chebyshev features T2(z) = 2 z^2 - 1
    T2 = 2.0 * (z**2) - 1.0
    cols.append(T2)

    # Pairwise interaction terms on the normalized z
    inters = []
    for i in range(F):
        for j in range(i+1, F):
            inters.append((z[:, i] * z[:, j])[:, None])
    if inters:
        cols.append(np.hstack(inters))

    # Concatenate into design matrix
    A = np.hstack(cols)  # shape (N, 1+4+4+6 = 15)
    return A

def scaling_law_func(data_points, params):
    """
    Predict LM loss via a 15‐parameter Chebyshev‐polynomial model in normalized log‐features:
      log_loss = A · θ
      loss     = exp(log_loss)

    Inputs:
      data_points: array‐like of shape (N,4) [lr, bsz, data_size, non_embedding_param_size]
      params:      array‐like of length 15

    Returns:
      preds: ndarray of shape (N,) of predicted LM losses
    """
    X = np.asarray(data_points, dtype=np.float64)
    # allow single‐point
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"scaling_law_func expects data_points shape (N,4), got {X.shape}")

    theta = np.asarray(params, dtype=np.float64).ravel()
    A = _build_design_matrix(X)  # (N,15)
    if theta.size != A.shape[1]:
        raise ValueError(f"scaling_law_func expects {A.shape[1]} parameters, got {theta.size}")

    log_pred = A.dot(theta)      # (N,)
    # Exponentiate to recover loss
    return np.exp(log_pred)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 15-parameter Chebyshev‐polynomial scaling law by ridge‐regularized
    least squares on log(loss):
      minimize ||A θ − log(y)||^2 + λ ||θ_{1:}||^2
    (No penalty on the intercept θ0.)

    Inputs:
      data_points: (N,4) array
      loss_values: (N,)   array

    Returns:
      theta_opt: ndarray of shape (15,) optimized parameters
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"fit_scaling_law expects data_points shape (N,4), got {X.shape}")

    y = np.asarray(loss_values, dtype=np.float64).ravel()
    if X.shape[0] != y.size:
        raise ValueError("fit_scaling_law: number of data points must match number of loss values")

    # Build design matrix and log‐target
    A = _build_design_matrix(X)    # (N,15)
    eps = 1e-12
    y_log = np.log(y + eps)        # (N,)

    # Normal equations for ridge regression
    ATA = A.T.dot(A)               # (15,15)
    ATy = A.T.dot(y_log)           # (15,)

    P = ATA.shape[0]
    # Adaptive regularization strength
    lam_base = 1e-3
    lam = lam_base * np.trace(ATA) / P
    # Build regularization matrix
    reg = lam * np.eye(P, dtype=np.float64)
    reg[0, 0] = 0.0  # do not penalize intercept

    ATA_reg = ATA + reg

    # If system is ill‐conditioned, increase regularization
    cond_threshold = 1e8
    cond_val = np.linalg.cond(ATA_reg)
    if cond_val > cond_threshold:
        # boost λ until condition number is acceptable
        for _ in range(3):
            lam *= 10.0
            reg = lam * np.eye(P, dtype=np.float64)
            reg[0, 0] = 0.0
            ATA_reg = ATA + reg
            cond_val = np.linalg.cond(ATA_reg)
            if cond_val <= cond_threshold:
                break

    # Solve robustly
    try:
        theta_opt = np.linalg.solve(ATA_reg, ATy)
    except np.linalg.LinAlgError:
        theta_opt = np.linalg.pinv(ATA_reg).dot(ATy)

    return theta_opt

# EVOLVE-BLOCK-END