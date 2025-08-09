import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Log‐linear scaling law:
      log(loss) ≈ a + b·log(tokens) + c·log(params) + d·log(unique_tokens)
    => loss = exp(a) * tokens^b * params^c * unique_tokens^d

    Inputs:
      data_points: array‐like of shape (N,3) [tokens, params, unique_tokens]
      params:      array‐like of length 4 [a, b, c, d]

    Returns:
      preds: shape (N,)
    """
    X = np.asarray(data_points, float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected data_points shape (N,3), got {X.shape}")
    p = np.asarray(params, float).ravel()
    if p.size != 4:
        raise ValueError(f"Expected 4 parameters [a, b, c, d], got {p.size}")
    a, b, c, d = p

    # Compute log inputs (epsilon to avoid log(0))
    logs = np.log(X + 1e-12)    # shape (N,3)
    # Linear model in log‐space
    y_log = a + b * logs[:, 0] + c * logs[:, 1] + d * logs[:, 2]
    # Back to original scale
    return np.exp(y_log)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the log‐linear scaling law by closed‐form ridge regression in log‐space.

    Returns:
      params_opt: array of 4 parameters [a, b, c, d]
    """
    X = np.asarray(data_points, float)
    y = np.asarray(loss_values, float).ravel()
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected data_points shape (N,3), got {X.shape}")

    # Use only strictly positive losses for log‐fit
    mask = y > 0
    if not np.any(mask):
        raise ValueError("All loss_values non‐positive; cannot fit a log‐linear model.")
    X, y = X[mask], y[mask]

    # Take logs (small epsilon to avoid -inf)
    logs = np.log(X + 1e-12)    # shape (n_samples,3)
    y_log = np.log(y + 1e-12)   # shape (n_samples,)

    # Design matrix for linear regression in log‐space: [1, log(tokens), log(params), log(unique_tokens)]
    n = logs.shape[0]
    D = np.empty((n, 4), float)
    D[:, 0] = 1.0
    D[:, 1:] = logs

    # Closed‐form ridge regression: minimize ||D·p − y_log||^2 + λ||p||^2
    G = D.T @ D
    # Auto‐tune regularization to data scale
    λ = 1e-6 * np.trace(G) / G.shape[0]
    # Solve (G + λI) p = D^T y_log
    coef = np.linalg.solve(G + np.eye(4) * λ, D.T @ y_log)

    return coef
# EVOLVE-BLOCK-END