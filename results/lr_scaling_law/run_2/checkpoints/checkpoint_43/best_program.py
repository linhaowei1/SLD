import numpy as np

# EVOLVE-BLOCK-START

def scaling_law_func(data_points, params):
    """
    Full quadratic scaling law in log-space with cross-terms.
    Features x = [lr, bsz, data_size, non_embedding_param_size].

    Model:
      log(loss) = θ0
                + Σ_i θ1+i       * log(x_i)
                + Σ_i θ1+F+i     * [log(x_i)]^2
                + Σ_{i<j} θ1+2F+idx(i,j) * log(x_i)*log(x_j)

    Returns loss = exp(predicted_log_loss).
    """
    X = np.asarray(data_points, dtype=float)      # shape (N,4)
    eps = 1e-12
    logX = np.log(X + eps)                        # safe log
    theta = np.asarray(params, dtype=float).ravel()
    N, F = logX.shape

    # Build design matrix Z column-wise:
    cols = []
    cols.append(np.ones(N))                       # intercept
    # linear terms
    for i in range(F):
        cols.append(logX[:, i])
    # quadratic terms
    for i in range(F):
        cols.append(logX[:, i]**2)
    # cross terms
    for i in range(F):
        for j in range(i+1, F):
            cols.append(logX[:, i] * logX[:, j])

    Z = np.stack(cols, axis=1)                    # shape (N, P)
    if Z.shape[1] != theta.size:
        raise ValueError(
            f"Parameter count mismatch: got {theta.size} params, "
            f"design matrix has {Z.shape[1]} columns."
        )

    pred_log = Z.dot(theta)
    return np.exp(pred_log)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the full quadratic-in-log model via ridge‐regularized least squares.

    Solves:
      minimize_θ ||Z θ - log(y)||^2 + λ * ||θ_{1:}||^2
      (no penalty on the intercept θ0).

    Returns the fitted θ vector.
    """
    X = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    eps = 1e-12

    # avoid log(0)
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)

    logX = np.log(X)
    logy = np.log(y)
    N, F = logX.shape

    # Build the same design matrix Z as in scaling_law_func
    cols = []
    cols.append(np.ones(N))                       # intercept
    for i in range(F):                            # linear
        cols.append(logX[:, i])
    for i in range(F):                            # quadratic
        cols.append(logX[:, i]**2)
    for i in range(F):                            # cross-terms
        for j in range(i+1, F):
            cols.append(logX[:, i] * logX[:, j])

    Z = np.stack(cols, axis=1)                    # shape (N, P)
    P = Z.shape[1]

    # Ridge‐regularization matrix (no penalty on intercept)
    lambda_reg = 1e-3
    I = np.eye(P, dtype=float)
    I[0, 0] = 0.0

    # Solve (Z^T Z + λ I) θ = Z^T logy
    A = Z.T.dot(Z) + lambda_reg * I
    b = Z.T.dot(logy)
    theta = np.linalg.solve(A, b)

    return theta

# EVOLVE-BLOCK-END