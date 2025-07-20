# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(proportions, params):
    """
    Domain‐mixture scaling law (7 params per domain):
      For each domain i:
        L_i = A_i 
             + B_i * log(p_i + eps) 
             + sum_{j=1..5} W_{i,j} * p_j
    params per domain i: [A_i, B_i, W_{i,1}, W_{i,2}, W_{i,3}, W_{i,4}, W_{i,5}]
    Total params = 5 domains × 7 = 35.
    """
    # Ensure array shape and clip for log stability
    P = np.atleast_2d(proportions).astype(float)
    eps = 1e-8
    P = np.clip(P, eps, 1.0)
    n_samples, n_dom = P.shape
    assert n_dom == 5, "Expect 5 domain proportions"
    
    # Prepare parameters
    p = np.array(params, dtype=float).flatten()
    if p.size < 35:
        p = np.concatenate([p, np.zeros(35 - p.size)])
    else:
        p = p[:35]
    p = p.reshape(5, 7)  # 5 domains × 7 params
    
    # Compute losses
    # Precompute log(p_i) terms
    logP = np.log(P)  # shape (n_samples, 5)
    L = np.zeros((n_samples, 5), dtype=float)
    
    # For each domain, compute A + B*log(p_i) + W·p_vector
    for i in range(5):
        A_i = p[i, 0]
        B_i = p[i, 1]
        W_i = p[i, 2:]          # shape (5,)
        # Broadcast and dot
        L[:, i] = A_i + B_i * logP[:, i] + P.dot(W_i)
    return L

def fit_scaling_law(proportions, loss_values):
    """
    Fit the above scaling law via regularized linear regression (ridge).
    For each domain i, solves:
      y_i = X_i · θ_i,  θ_i = [A_i, B_i, W_{i,1..5}]
    where X_i columns are [1, log(p_i), p1..p5].
    """
    P = np.atleast_2d(proportions).astype(float)
    Y = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = P.shape
    assert n_dom == 5 and Y.shape == (n_samples, 5)
    
    eps = 1e-8
    P_clipped = np.clip(P, eps, 1.0)
    logP = np.log(P_clipped)
    
    # Build design matrix common parts
    # We will rebuild per-domain because log(p_i) column differs
    params = np.zeros((5, 7), dtype=float)
    
    # Regularization strength
    alpha = 1e-3
    
    for i in range(5):
        # X_i: [1, log(p_i), p1, p2, p3, p4, p5]
        Xi = np.concatenate([
            np.ones((n_samples, 1)),
            logP[:, [i]],
            P_clipped
        ], axis=1)  # shape (n_samples, 7)
        
        yi = Y[:, i]  # shape (n_samples,)
        
        # Solve ridge: (X^T X + alpha I) θ = X^T y
        XtX = Xi.T.dot(Xi)
        reg = alpha * np.eye(7, dtype=float)
        A = XtX + reg
        b = Xi.T.dot(yi)
        
        # Solve for theta
        theta_i = np.linalg.solve(A, b)
        params[i, :] = theta_i
    
    return params.flatten()

# declare expected param count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END