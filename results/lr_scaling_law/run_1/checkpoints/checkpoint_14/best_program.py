# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Predict lm loss using an additive‐bias power‐law model:
      loss ≈ b + exp(logA) * lr^{e_lr} * bsz^{e_bsz} * data_size^{e_data} * model_size^{e_model}

    data_points: (N,4) array [lr, bsz, data_size, non_embedding_param_size]
    params:      (6,) array [b, logA, e_lr, e_bsz, e_data, e_model]
    returns:     (N,) array of predicted losses
    """
    X = np.asarray(data_points, dtype=float)
    # Ensure shape (N,4)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 4:
        raise ValueError(f"Expected data_points with 4 columns, got {X.shape[1]}")
    b, logA, e_lr, e_bsz, e_data, e_model = params
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    lr  = X[:, 0] + eps
    bsz = X[:, 1] + eps
    D   = X[:, 2] + eps
    Np  = X[:, 3] + eps
    # Compute power-law term in log‐space for numerical stability
    log_term = logA \
             + e_lr  * np.log(lr) \
             + e_bsz * np.log(bsz) \
             + e_data * np.log(D) \
             + e_model* np.log(Np)
    return b + np.exp(log_term)

def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter model to minimize MSE:
      params = [b, logA, e_lr, e_bsz, e_data, e_model]

    Uses a closed‐form log‐linear regression to initialize exponents,
    then refines via L-BFGS-B with bounded exponents for stability.
    Returns: optimized (6,) parameter array.
    """
    # Prepare data
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float).ravel()
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != 4:
        raise ValueError(f"Expected data_points with 4 columns, got {X.shape[1]}")

    # 1) Closed-form log‐linear regression (no additive bias)
    eps = 1e-12
    Z = np.log(X + eps)                      # shape (N,4)
    A = np.column_stack([np.ones(len(y)), Z])  # design matrix (N,5)
    y_log = np.log(y + eps)                  # shape (N,)
    theta_ls, *_ = np.linalg.lstsq(A, y_log, rcond=None)
    # theta_ls = [logA_init, e_lr_init, e_bsz_init, e_data_init, e_model_init]

    # 2) Build initial parameter vector: b=0, rest from LS
    init_params = np.zeros(6, dtype=float)
    init_params[0] = 0.0            # b
    init_params[1] = theta_ls[0]    # logA
    init_params[2:] = theta_ls[1:]  # [e_lr, e_bsz, e_data, e_model]

    # 3) Define bounds for stability
    y_max = np.max(y)
    bounds = [
        (0.0, max(y_max, 1.0)),  # b >= 0
        (None, None),            # logA free
        (-5.0, 5.0),             # e_lr
        (-5.0, 5.0),             # e_bsz
        (-5.0, 5.0),             # e_data
        (-5.0, 5.0),             # e_model
    ]

    # 4) Objective: mean squared error in original loss space
    def mse_obj(p):
        pred = scaling_law_func(X, p)
        return np.mean((pred - y) ** 2)

    # 5) Optimize with L-BFGS-B
    result = minimize(
        mse_obj,
        init_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    # Return optimized params or fallback to initial
    return result.x if result.success else init_params
# EVOLVE-BLOCK-END