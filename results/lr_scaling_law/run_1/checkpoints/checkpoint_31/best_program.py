import numpy as np

# Pre‐defined log‐ranges for normalization (from problem description)
_LOG_MIN = np.log(np.array([1.2e-4,    # lr
                            16.0,      # bsz
                            4e9,       # data_size
                            2.14e8     # non_embedding_param_size
                           ], dtype=np.float64))
_LOG_MAX = np.log(np.array([2.2e-2,    # lr
                            4096.0,    # bsz
                            1e11,      # data_size
                            1e9        # non_embedding_param_size
                           ], dtype=np.float64))
_LOG_MID = 0.5 * (_LOG_MIN + _LOG_MAX)
_LOG_HALF_RANGE = 0.5 * (_LOG_MAX - _LOG_MIN)


def scaling_law_func(data_points, params):
    """
    Full quadratic scaling law in normalized log‐space including pairwise interactions:
      z_i = (log(x_i) - mid_i) / half_range_i
      log_loss = θ0
               + sum_i θ1_i * z_i
               + sum_i θ2_i * z_i^2
               + sum_{i<j} θ3_{ij} * z_i * z_j
      loss = exp(log_loss)
    params is a vector of length 1 + 4 + 4 + 6 = 15:
      [θ0,
       θ1_lr, θ1_bsz, θ1_data, θ1_param,
       θ2_lr, θ2_bsz, θ2_data, θ2_param,
       θ3_lr_bsz, θ3_lr_data, θ3_lr_param,
       θ3_bsz_data, θ3_bsz_param,
       θ3_data_param]
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected 4 features, got {F}")
    p = np.asarray(params, dtype=np.float64).ravel()
    expected_len = 1 + F + F + (F * (F - 1) // 2)
    if p.size != expected_len:
        raise ValueError(f"Expected params of length {expected_len}, got {p.size}")

    # Unpack parameters
    theta0 = p[0]
    lin_coeffs = p[1 : 1 + F]
    quad_coeffs = p[1 + F : 1 + 2 * F]
    cross_coeffs = p[1 + 2 * F :]

    # Compute normalized logs
    eps = 1e-12
    logs = np.log(X + eps)           # shape (N,4)
    z = (logs - _LOG_MID) / _LOG_HALF_RANGE

    # Build log‐prediction
    log_pred = theta0 + z.dot(lin_coeffs) + (z ** 2).dot(quad_coeffs)

    # Add pairwise interaction terms
    k = 0
    for i in range(F):
        for j in range(i + 1, F):
            log_pred += cross_coeffs[k] * (z[:, i] * z[:, j])
            k += 1

    # Return in original loss space
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 15‐parameter quadratic‐in‐normalized‐log scaling law by
    regularized least squares on log(loss).
    Returns the parameter vector of length 15.
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    y = np.asarray(loss_values, dtype=np.float64).ravel()

    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected 4 features, got {F}")
    if y.shape[0] != N:
        raise ValueError("Number of data points and losses must match")

    # Build design matrix in normalized log‐space
    eps = 1e-12
    logs = np.log(X + eps)
    z = (logs - _LOG_MID) / _LOG_HALF_RANGE

    # Intercept
    A_cols = [np.ones((N, 1), dtype=np.float64)]
    # Linear terms
    A_cols.append(z)
    # Quadratic terms
    A_cols.append(z ** 2)
    # Pairwise interaction terms
    cross_cols = []
    for i in range(F):
        for j in range(i + 1, F):
            cross_cols.append((z[:, i] * z[:, j])[:, None])
    if cross_cols:
        A_cols.append(np.concatenate(cross_cols, axis=1))

    A = np.concatenate(A_cols, axis=1)  # shape (N,15)

    # Target in log‐space
    y_log = np.log(y + eps)

    # Regularized normal equations: (A^T A + λI) θ = A^T y_log
    lam = 1e-5
    ATA = A.T.dot(A)
    ATA_reg = ATA + lam * np.eye(ATA.shape[0])
    ATy = A.T.dot(y_log)

    theta = np.linalg.solve(ATA_reg, ATy)
    return theta