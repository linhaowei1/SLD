# EVOLVE-BLOCK-START
"""
Improved domain‐mixture scaling law using a linear model in negative log‐proportions.
Each domain loss Li is modeled as:
    Li = α_i + ∑_{j=1..5} β_{ij} * ( -log(p_j + ε) )
Total parameters per domain: 1 (α_i) + 5 (β_{ij}) = 6. 
Overall parameters: 5 domains × 6 = 30.
This yields a simple, numerically stable closed‐form fit via least squares.
"""

import numpy as np

def scaling_law_func(proportions, params):
    """
    Predict per‐domain losses from domain mixture proportions.
    
    Args:
        proportions: array [n_samples, 5], each row sums to 1.0
        params:      array of length 30 (5 domains × 6 params each)
                     ordered as [ (α_1, β_1,1...β_1,5),
                                   (α_2, β_2,1...β_2,5),
                                    ...
                                   (α_5, β_5,1...β_5,5) ]
    Returns:
        predicted_losses: array [n_samples, 5]
    """
    proportions = np.atleast_2d(proportions)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5, "Expected 5 domain proportions per sample"
    
    params = np.ravel(params)
    assert params.size == 30, f"Expected 30 parameters, got {params.size}"
    # Reshape to (5 domains, 6 params each)
    P = params.reshape(5, 6)  
    alphas = P[:, 0]         # shape (5,)
    betas  = P[:, 1:]        # shape (5,5)
    
    # Compute features: negative log‐proportions
    eps = 1e-8
    nl = -np.log(proportions + eps)   # shape (n_samples, 5)
    
    # Predict: for each domain i, Li = α_i + ∑_j β_{i,j} * nl[:, j]
    # => predicted = nl @ betas.T + alphas
    predicted = nl.dot(betas.T) + alphas[np.newaxis, :]
    return predicted

def fit_scaling_law(proportions, loss_values):
    """
    Fit the linear negative‐log proportion model by solving 5 independent least squares.
    
    Args:
        proportions: array [n_samples, 5]
        loss_values: array [n_samples, 5]
    Returns:
        params: array length 30, optimized parameters for scaling_law_func
    """
    proportions = np.atleast_2d(proportions)
    loss_values = np.atleast_2d(loss_values)
    n_samples, n_dom = proportions.shape
    assert n_dom == 5 and loss_values.shape == (n_samples, 5)
    
    # Build design matrix X: [ones, -log(proportions)]
    eps = 1e-8
    neg_log = -np.log(proportions + eps)        # (n_samples, 5)
    X = np.concatenate([np.ones((n_samples, 1)), neg_log], axis=1)  # (n_samples, 6)
    
    # Solve per‐domain least squares: minimize ||X·θ - y||^2
    all_params = []
    for i in range(5):
        y = loss_values[:, i]
        θ, *_ = np.linalg.lstsq(X, y, rcond=None)  # θ shape (6,)
        all_params.append(θ)
    
    # Stack and flatten to length 30
    fitted = np.vstack(all_params)  # (5,6)
    return fitted.ravel()

# Expose expected parameter count
scaling_law_func.num_params = 30

# EVOLVE-BLOCK-END