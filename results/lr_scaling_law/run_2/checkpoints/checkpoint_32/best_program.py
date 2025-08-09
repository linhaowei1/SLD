import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Flexible scaling law in log‐space supporting:
      - pure power‐law:           params length = 1 + F
      - quadratic log‐law:        params length = 1 + 2F
      - full quadratic w/ crosses: params length = 1 + 2F + F*(F-1)/2
    where F = number of features (here 4).
    """
    X = np.asarray(data_points, dtype=float)
    eps = 1e-12
    # log‐transform inputs
    logX = np.log(X + eps)
    theta = np.asarray(params, dtype=float).ravel()
    N, F = logX.shape
    P = theta.size

    # Build design matrix Z based on expected param length
    # Start with intercept and linear terms
    Z_parts = [np.ones((N, 1), dtype=float), logX]

    # If quadratic terms are present
    if P >= 1 + 2*F:
        Z_parts.append(logX**2)

    # If cross‐terms are present
    full_cross_len = (1 + 2*F + F*(F-1)//2)
    if P == full_cross_len:
        # add pairwise product of distinct features
        for i in range(F):
            for j in range(i+1, F):
                Z_parts.append((logX[:, i] * logX[:, j]).reshape(N, 1))

    Z = np.hstack(Z_parts)
    if Z.shape[1] != P:
        raise ValueError(f"Parameter vector of length {P} "
                         f"does not match design matrix width {Z.shape[1]}.")

    # Predict in log‐space and exponentiate
    pred_log = Z.dot(theta)
    return np.exp(pred_log)


def fit_scaling_law(data_points, loss_values):
    """
    Fits a full quadratic log‐law with cross‐terms:
       log(loss) ≈ intercept
                 + Σ_i a_i·log(x_i)
                 + Σ_i b_i·[log(x_i)]^2
                 + Σ_{i<j} c_{ij}·log(x_i)·log(x_j)
    via ridge‐regularized least squares in log‐space.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    eps = 1e-12

    # log‐transform
    logX = np.log(X + eps)
    logy = np.log(y + eps)
    N, F = logX.shape

    # Determine full‐model parameter count
    P = 1 + 2*F + (F*(F-1)//2)

    # Build design matrix
    Z_parts = [np.ones((N, 1), dtype=float), logX, logX**2]
    for i in range(F):
        for j in range(i+1, F):
            Z_parts.append((logX[:, i] * logX[:, j]).reshape(N, 1))
    Z = np.hstack(Z_parts)  # shape (N, P)

    # Ridge regularization (no penalty on intercept)
    lambda_reg = 1e-3
    reg = np.eye(P, dtype=float)
    reg[0, 0] = 0.0

    # Solve normal equations: (Z^T Z + λ I) θ = Z^T logy
    A = Z.T.dot(Z) + lambda_reg * reg
    b = Z.T.dot(logy)
    theta = np.linalg.solve(A, b)

    return theta
# EVOLVE-BLOCK-END