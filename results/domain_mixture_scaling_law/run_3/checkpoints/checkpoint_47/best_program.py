"""
Scaling law discovery for LLM multi-domain loss prediction
Using a hybrid linear + log‐feature mixture‐of‐experts model:
  preds_j = X @ W_lin[:,j]  +  w_log[j] * log(X[:,j] + ε)  +  b[j]
Total parameters = 5×5 (W_lin) + 5 (w_log) + 5 (b) = 35 ≤ 35.
Fitting is done in closed form with ridge regularization for stability.
"""
import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Predict multi-domain losses given domain mixture proportions,
    using a combination of linear and per-domain log features.

    preds_{n,j} = sum_i X[n,i] * W_lin[i,j]
                + w_log[j] * log(X[n,j] + eps)
                + b[j]

    Args:
        data_points: array of shape (N,5) with domain mixture proportions.
        params:      flat array of length 35:
                     - first 25 entries → W_lin of shape (5,5)
                     - next 5 entries   → w_log of shape (5,)
                     - last 5 entries   → b      of shape (5,)

    Returns:
        preds: array of shape (N,5), predicted losses per domain.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"data_points must have shape (N,5), got {X.shape}")
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 35:
        raise ValueError(f"Expected 35 parameters, got {p.size}")

    # unpack
    W_lin = p[:25].reshape(5, 5)     # linear weights
    w_log = p[25:30].reshape(5,)     # log-feature weights
    b     = p[30:35].reshape(1, 5)   # biases

    # linear part
    out_lin = X.dot(W_lin)           # shape (N,5)

    # log part (per-domain)
    eps = 1e-8
    X_log = np.log(X + eps)          # shape (N,5)
    out_log = X_log * w_log          # broadcasts (N,5)

    preds = out_lin + out_log + b    # shape (N,5)
    return preds


def fit_scaling_law(data_points, loss_values):
    """
    Fit the hybrid linear+log scaling law by solving 5 separate
    ridge‐regularized least squares problems (one per output domain).

    Args:
        data_points: array of shape (N,5) with domain proportions.
        loss_values: array of shape (N,5) of observed multi-domain losses.

    Returns:
        params: flat array of length 35 (25 W_lin + 5 w_log + 5 b).
    """
    X = np.asarray(data_points, dtype=float)
    Y = np.asarray(loss_values, dtype=float)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"data_points must have shape (N,5), got {X.shape}")
    if Y.ndim != 2 or Y.shape != X.shape:
        raise ValueError(f"loss_values must have shape {X.shape}, got {Y.shape}")

    N = X.shape[0]
    eps = 1e-8
    X_log = np.log(X + eps)           # (N,5)

    # We'll fit each output j separately:
    W_lin = np.zeros((5, 5), dtype=float)
    w_log = np.zeros(5, dtype=float)
    b_vec = np.zeros(5, dtype=float)

    lam = 1e-6  # ridge regularization strength
    # Prepare identity for ridge
    # For each j, design matrix has 7 cols: [X[:,0..4], X_log[:,j], 1]
    I7 = np.eye(7, dtype=float)

    for j in range(5):
        # Build design matrix for target j
        # features: X (5 cols), X_log[:,j] (1 col), ones (1 col) → total 7
        Xj = np.concatenate([
            X,                          # (N,5)
            X_log[:, j:j+1],            # (N,1)
            np.ones((N, 1), dtype=float)# (N,1)
        ], axis=1)  # shape (N,7)

        yj = Y[:, j]  # shape (N,)

        # Solve (Xj^T Xj + lam I) θ = Xj^T yj
        A = Xj.T.dot(Xj) + lam * I7    # (7,7)
        B = Xj.T.dot(yj)               # (7,)
        theta_j = np.linalg.solve(A, B)  # (7,)

        # Unpack
        W_lin[:, j] = theta_j[0:5]
        w_log[j]    = theta_j[5]
        b_vec[j]    = theta_j[6]

    # Flatten into parameter vector
    params = np.concatenate([W_lin.ravel(), w_log, b_vec])
    return params

# EVOLVE-BLOCK-END