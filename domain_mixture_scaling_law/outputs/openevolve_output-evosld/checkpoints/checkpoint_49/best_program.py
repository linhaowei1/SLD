import numpy as np

# =============================================================================
# EVOLVE-BLOCK-START
# =============================================================================

def scaling_law_func(proportions, params):
    """
    Domain‐mixture scaling law (7 params per domain, 35 total):
      For domain i:
        L_i = A_i 
              + B_i * log(p_i + eps)
              + C_i * sqrt(p_i + eps)
              + sum_{j != i} W_{i,j} * p_j

    params: flat array length ≤35. Interpreted as 5×7 matrix:
      [A_i, B_i, C_i, W_i,j1, W_i,j2, W_i,j3, W_i,j4] for each i in 0..4
    """
    P = np.asarray(proportions, float)
    # ensure shape (n_samples, 5)
    if P.ndim == 1:
        P = P[np.newaxis, :]
    n_samples, n_dom = P.shape
    assert n_dom == 5, "Expect 5 domain proportions"

    # numerical safety
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    logP = np.log(P)
    sqrtP = np.sqrt(P)

    # unpack params into (5,7)
    p = np.ravel(params).astype(float)
    if p.size < 35:
        p = np.concatenate([p, np.zeros(35 - p.size)])
    else:
        p = p[:35]
    p = p.reshape(5, 7)

    # build mask for "other domains"
    mask = ~np.eye(5, dtype=bool)

    # compute losses
    L = np.empty((n_samples, 5), dtype=float)
    for i in range(5):
        A_i, B_i, C_i = p[i, 0], p[i, 1], p[i, 2]
        W_i = p[i, 3:]              # weights for the 4 other domains
        P_other = P[:, mask[i]]     # shape (n_samples, 4)
        L[:, i] = A_i \
                  + B_i * logP[:, i] \
                  + C_i * sqrtP[:, i] \
                  + P_other.dot(W_i)
    return L


def fit_scaling_law(proportions, loss_values, alpha=1e-6):
    """
    Fit the 7-param scaling law per domain by ridge regression.

    Returns flat params array of length 35.
    """
    P = np.asarray(proportions, float)
    Y = np.asarray(loss_values, float)
    # ensure 2D
    if P.ndim == 1:
        P = P[np.newaxis, :]
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]
    n_samples, n_dom = P.shape
    assert n_dom == 5 and Y.shape == (n_samples, 5)

    # numerical safety
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    logP = np.log(P)
    sqrtP = np.sqrt(P)

    # precompute mask and identity
    mask = ~np.eye(5, dtype=bool)
    I7 = np.eye(7, dtype=float)

    # container for parameters
    params = np.zeros((5, 7), dtype=float)

    # solve per-domain ridge regression
    for i in range(5):
        P_other = P[:, mask[i]]  # shape (n_samples, 4)
        # design matrix: [1, log(p_i), sqrt(p_i), p_other_j...]
        Xi = np.column_stack([
            np.ones(n_samples),
            logP[:, i],
            sqrtP[:, i],
            P_other
        ])  # (n_samples, 7)
        yi = Y[:, i]  # target
        # closed-form ridge: θ = (XᵀX + αI)⁻¹ Xᵀ y
        XtX = Xi.T.dot(Xi)
        Xty = Xi.T.dot(yi)
        params[i, :] = np.linalg.solve(XtX + alpha * I7, Xty)

    return params.ravel()


# expose expected parameter count
scaling_law_func.num_params = 35

# =============================================================================
# EVOLVE-BLOCK-END
# =============================================================================