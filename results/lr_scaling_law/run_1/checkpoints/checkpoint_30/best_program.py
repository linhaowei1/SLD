import numpy as np

def scaling_law_func(data_points, params):
    """
    Predicts LM loss from hyperparameters via an extended multiplicative power‐law model:
      log(loss) ≈ intercept + Σ_i w_i * log(x_i) + Σ_i v_i * (log(x_i))^2
    where x = [lr, bsz, data_size, non_embedding_param_size].
    
    Inputs:
      data_points: array of shape (N,4)
      params:       array of shape (1 + 4 + 4,) = [intercept, w1..w4, v1..v4]
                    or shape (T, 9) for multi‐target
    Returns:
      preds: array of shape (N,) (or (N,T) if multi‐target)
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    eps = 1e-12
    # avoid log(0)
    X = np.maximum(X, eps)
    # compute log‐features
    logX = np.log(X)                     # (N,4)
    lin = logX                          # (N,4)
    quad = logX * logX                  # (N,4)
    
    N, F = logX.shape                   # F == 4
    # assemble design matrix Z = [1, lin, quad]
    Z = np.concatenate([np.ones((N,1), dtype=np.float64), lin, quad], axis=1)  # (N,9)
    
    theta = np.asarray(params, dtype=np.float64)
    # support multi‐target: ensure shape (T,P)
    if theta.ndim == 1:
        theta = theta[None, :]
    T, P = theta.shape
    if P != 1 + 2*F:
        raise ValueError(f"Expected parameter length {1+2*F}, got {P}")
    
    # predictive log‐loss, then exponentiate
    pred_log = Z.dot(theta.T)           # (N,T)
    pred = np.exp(pred_log)             # (N,T)
    
    # flatten if single‐target
    return pred.ravel() if T == 1 else pred


def fit_scaling_law(data_points, loss_values):
    """
    Fits the extended power‐law model by ridge‐regularized linear regression
    in log‐space with quadratic terms:
      log(loss) ≈ intercept + Σ_i w_i * log(x_i) + Σ_i v_i * (log(x_i))^2
    
    Returns params of shape (1 + 4 + 4,) = [intercept, w1..w4, v1..v4].
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=np.float64))
    y = np.asarray(loss_values, dtype=np.float64)
    # avoid zeros
    eps = 1e-12
    X = np.maximum(X, eps)
    y = np.maximum(y, eps)
    
    logX = np.log(X)                    # (N,4)
    logy = np.log(y)                    # (N,)
    N, F = logX.shape                   # F == 4
    
    # design matrix Z = [1, logX, logX^2]
    lin = logX
    quad = logX * logX
    Z = np.concatenate([np.ones((N,1), dtype=np.float64), lin, quad], axis=1)  # (N,1+2F)
    P = Z.shape[1]
    
    # set up ridge regularization: no penalty on intercept,
    # small λ on linear terms, slightly larger on quadratic
    lambda_lin = 1e-6
    lambda_quad = 1e-4
    reg = np.zeros((P, P), dtype=np.float64)
    # indices 1..F  ← λ_lin
    for i in range(1, 1+F):
        reg[i, i] = lambda_lin
    # indices (1+F)..(1+2F-1)  ← λ_quad
    for i in range(1+F, P):
        reg[i, i] = lambda_quad
    
    # normal equations: (Z^T Z + reg) θ = Z^T logy
    A = Z.T.dot(Z) + reg
    b = Z.T.dot(logy)
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback if singular
        theta, *_ = np.linalg.lstsq(Z, logy, rcond=None)
    
    return theta