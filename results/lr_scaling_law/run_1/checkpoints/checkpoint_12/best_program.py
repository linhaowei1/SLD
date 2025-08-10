# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(data_points, params):
    """
    Quadratic‐in‐log scaling law:
      Loss = exp( θ0 
                  + sum_i θ1_i * log(x_i)
                  + sum_i θ2_i * (log(x_i))^2 )
    where x_i ∈ {lr, bsz, data_size, non_embed_param_size}.
    params: length = 1 + 2*4 = 9
      [θ0, θ1_lr, θ1_bsz, θ1_data, θ1_param,
           θ2_lr, θ2_bsz, θ2_data, θ2_param]
    """
    X = np.asarray(data_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    N, F = X.shape
    if F != 4:
        raise ValueError(f"Expected 4 features, got {F}")
    p = np.asarray(params, dtype=np.float64).ravel()
    expected_len = 1 + 2 * F
    if p.size != expected_len:
        raise ValueError(f"Expected params of length {expected_len}, got {p.size}")
    theta0 = p[0]
    lin_coeffs = p[1:1+F]
    quad_coeffs = p[1+F:1+2*F]

    # numerical stability
    eps = 1e-12
    logs = np.log(X + eps)
    log_pred = theta0 \
               + logs.dot(lin_coeffs) \
               + (logs**2).dot(quad_coeffs)
    return np.exp(log_pred)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 9 parameters by least‐squares on log(Loss):
      log(y) ≃ θ0 
               + sum_i θ1_i log(x_i)
               + sum_i θ2_i (log(x_i))^2
    Returns the parameter vector of length 9.
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

    # Build design matrix with intercept, log, and log^2 terms
    eps = 1e-12
    logs = np.log(X + eps)            # (N,4)
    logs2 = logs**2                   # (N,4)
    A = np.concatenate([
        np.ones((N, 1), dtype=np.float64),
        logs,
        logs2
    ], axis=1)                        # (N, 1+4+4)

    y_log = np.log(y + eps)

    # Regularized normal equations: (A^T A + λI) θ = A^T y_log
    lam = 1e-6
    ATA = A.T.dot(A)
    ATA_reg = ATA + lam * np.eye(ATA.shape[0])
    ATy = A.T.dot(y_log)

    theta = np.linalg.solve(ATA_reg, ATy)
    return theta
# EVOLVE-BLOCK-END