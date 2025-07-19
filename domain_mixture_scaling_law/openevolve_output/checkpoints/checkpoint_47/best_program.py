# =============================================================================
# EVOLVE-BLOCK-START
# =============================================================================
import numpy as np

def scaling_law_func(proportions, params):
    """
    Enhanced domain‐mixture scaling law (7 params per domain):
      For each domain i:
        L_i = A_i 
             + B_i * log(p_i + eps)
             + C_i * sqrt(p_i + eps)
             + sum_{j != i} W_{i,j} * p_j

    params per domain i: [A_i, B_i, C_i, W_i,0, W_i,1, W_i,2, W_i,3, W_i,4]
      (W_i,i is present in the array but not used in the sum)
    Total params = 5 domains × 7 = 35.
    """
    P = np.atleast_2d(proportions).astype(float)
    eps = 1e-12
    P = np.clip(P, eps, 1.0)      # avoid log(0) or division issues
    n_samples, n_dom = P.shape
    assert n_dom == 5, "Expect 5 domain proportions"

    # Flatten/pad params to (5×7)
    p = np.asarray(params, dtype=float).flatten()
    if p.size < 35:
        p = np.concatenate([p, np.zeros(35 - p.size)])
    else:
        p = p[:35]
    p = p.reshape(5, 7)

    logP = np.log(P)
    sqrtP = np.sqrt(P)
    L = np.zeros((n_samples, 5), dtype=float)

    # compute each domain loss
    for i in range(5):
        A_i, B_i, C_i = p[i, 0], p[i, 1], p[i, 2]
        W_i = p[i, 3:]  # length 4–5, but we will only dot with p_j, j!=i

        # build p_other by removing column i
        mask = np.ones(5, bool)
        mask[i] = False
        P_other = P[:, mask]       # shape (n_samples, 4)

        # L_i = A + B*log(p_i) + C*sqrt(p_i) + W_i·p_other
        L[:, i] = (
            A_i
            + B_i * logP[:, i]
            + C_i * sqrtP[:, i]
            + P_other.dot(W_i)
        )

    return L


def fit_scaling_law(proportions, loss_values):
    """
    Fit the enhanced scaling law via ridge‐regularized linear regression.
    For each domain i, regress:
      y_i = A_i 
            + B_i * log(p_i)
            + C_i * sqrt(p_i)
            + sum_{j != i} W_{i,j} * p_j

    Regressors per domain: [1, log(p_i), sqrt(p_i), p_j for j!=i] → 7 columns
    """
    P = np.atleast_2d(proportions).astype(float)
    Y = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = P.shape
    assert n_dom == 5 and Y.shape == (n_samples, 5)

    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    logP = np.log(P)
    sqrtP = np.sqrt(P)

    alpha = 1e-3  # ridge regularization
    params = np.zeros((5, 7), dtype=float)

    for i in range(5):
        # build matrix of cross‐domain proportions (exclude domain i)
        mask = np.ones(5, bool)
        mask[i] = False
        P_other = P[:, mask]  # shape (n_samples, 4)

        # design matrix: [1, log(p_i), sqrt(p_i), p_other...]
        Xi = np.hstack([
            np.ones((n_samples, 1)),
            logP[:, [i]],
            sqrtP[:, [i]],
            P_other
        ])  # shape (n_samples, 7)

        yi = Y[:, i]  # target losses for domain i

        # solve ridge: (X^T X + αI) θ = X^T y
        XtX = Xi.T.dot(Xi)
        A_mat = XtX + alpha * np.eye(7)
        b_vec = Xi.T.dot(yi)
        theta_i = np.linalg.solve(A_mat, b_vec)

        params[i, :] = theta_i

    return params.flatten()


# expose expected parameter count
scaling_law_func.num_params = 35
# =============================================================================
# EVOLVE-BLOCK-END
# =============================================================================