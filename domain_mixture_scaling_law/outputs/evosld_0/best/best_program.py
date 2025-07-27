import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(proportions, params):
    """
    Predict per‐domain losses given mixture proportions and linear‐feature parameters.
    Functional form per domain i (7 parameters):
        Li = w0_i
           + w1_i * (-log p_i)
           + w2_i * (-log(1 - p_i))
           + w3_i * p_i
           + w4_i * (1 - p_i)
           + w5_i * H
           + w6_i * G

    where p_i is clipped to [eps,1-eps], H = -sum_j p_j log p_j, G = 1 - sum_j p_j^2.
    params is a flat array of length up to 35 (5 domains × 7).
    """
    P = np.atleast_2d(proportions).astype(float)
    n, d = P.shape
    assert d == 5, "Expected 5 domains in proportions"
    # Clip for numerical stability
    eps = 1e-8
    P = np.clip(P, eps, 1.0 - eps)
    # Cross‐domain features
    H = -np.sum(P * np.log(P), axis=1)  # shape [n]
    G = 1.0 - np.sum(P * P, axis=1)      # shape [n]
    # Prepare parameter matrix
    flat = np.ravel(params).astype(float)
    total = 5 * 7
    if flat.size < total:
        buf = np.zeros(total, dtype=float)
        buf[: flat.size] = flat
        flat = buf
    else:
        flat = flat[:total]
    W = flat.reshape(5, 7)  # each row: [w0, w1, ..., w6]
    # Build features and compute losses
    ones = np.ones((n, 1))
    xp = -np.log(P)             # shape [n,5]
    xcp = -np.log(1.0 - P)      # shape [n,5]
    H_col = H[:, None]          # shape [n,1]
    G_col = G[:, None]          # shape [n,1]

    losses = np.zeros((n, 5), dtype=float)
    for i in range(5):
        Xi = np.concatenate([
            ones,
            xp[:, i : i + 1],
            xcp[:, i : i + 1],
            P[:, i : i + 1],
            (1.0 - P[:, i : i + 1]),
            H_col,
            G_col
        ], axis=1)  # [n,7]
        losses[:, i] = Xi.dot(W[i])
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the linear‐feature scaling law by closed‐form least squares.
    Returns optimized flat parameter array of length 35.
    """
    P = np.atleast_2d(proportions).astype(float)
    L = np.atleast_2d(loss_values).astype(float)
    n, d = P.shape
    assert d == 5 and L.shape == (n, 5), "Expected shapes (n,5)"
    # Clip and compute shared features
    eps = 1e-8
    P = np.clip(P, eps, 1.0 - eps)
    H = -np.sum(P * np.log(P), axis=1)
    G = 1.0 - np.sum(P * P, axis=1)
    ones = np.ones((n, 1))
    xp = -np.log(P)
    xcp = -np.log(1.0 - P)
    H_col = H[:, None]
    G_col = G[:, None]

    all_params = []
    for i in range(5):
        Xi = np.concatenate([
            ones,
            xp[:, i : i + 1],
            xcp[:, i : i + 1],
            P[:, i : i + 1],
            (1.0 - P[:, i : i + 1]),
            H_col,
            G_col
        ], axis=1)  # [n,7]
        # closed‐form least squares
        w_i, *_ = np.linalg.lstsq(Xi, L[:, i], rcond=None)
        all_params.append(w_i)
    all_params = np.vstack(all_params)  # [5,7]
    return all_params.ravel()

# expose parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END