"""
Human-designed Data-Constrained Scaling Law
L(N,D,U) = E + A/((U_N + U_N*R_N*(1-exp(-(N/U_N-1)/R_N)))^α) + B/((U + U*R_D*(1-exp(-(D/U-1)/R_D)))^β)
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Human-designed data-constrained scaling law function.
    
    L(N,D,U) = E + A/((U_N + U_N*R_N*(1-exp(-(N/U_N-1)/R_N)))^α) + B/((U + U*R_D*(1-exp(-(D/U-1)/R_D)))^β)
    
    Simplified version using 7 parameters: [E, A, α, B, β, R_N, R_D]
    We assume U_N = model_size and U = unique_tokens for simplicity
    
    Args:
        tokens: Array of training tokens used (D)
        model_size: Array of model parameter counts (N)
        unique_tokens: Array of unique tokens available (U)
        params: Array of parameters [E, A, α, B, β, R_N, R_D] (7 parameters)
        
    Returns:
        Predicted loss values
    """
    E, A, alpha, B, beta, R_N, R_D = params
    
    # Convert to numpy arrays and ensure positive values
    N = np.asarray(model_size, dtype=float) + 1e-8
    D = np.asarray(tokens, dtype=float) + 1e-8
    U = np.asarray(unique_tokens, dtype=float) + 1e-8
    
    # For simplicity, use U_N = N and U = U
    U_N = N
    
    # Ensure R_N and R_D are positive
    R_N = max(abs(R_N), 1e-8)
    R_D = max(abs(R_D), 1e-8)
    
    # Calculate the first term: A/((U_N + U_N*R_N*(1-exp(-(N/U_N-1)/R_N)))^α)
    ratio_N = (N / U_N - 1) / R_N
    ratio_N = np.clip(ratio_N, -50, 50)  # Clip to avoid overflow
    exp_term_N = 1 - np.exp(-ratio_N)
    denominator_N = U_N + U_N * R_N * exp_term_N
    denominator_N = np.maximum(denominator_N, 1e-8)
    first_term = A / np.power(denominator_N, alpha)
    
    # Calculate the second term: B/((U + U*R_D*(1-exp(-(D/U-1)/R_D)))^β)
    ratio_D = (D / U - 1) / R_D
    ratio_D = np.clip(ratio_D, -50, 50)  # Clip to avoid overflow
    exp_term_D = 1 - np.exp(-ratio_D)
    denominator_D = U + U * R_D * exp_term_D
    denominator_D = np.maximum(denominator_D, 1e-8)
    second_term = B / np.power(denominator_D, beta)
    
    # Combine terms
    loss = E + first_term + second_term
    
    return loss

# EVOLVE-BLOCK-START
def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the human-designed data-constrained scaling law to data using multi-start L-BFGS-B optimization.
    
    Args:
        tokens: Array of training tokens used
        model_size: Array of model parameter counts
        unique_tokens: Array of unique tokens available
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (7 parameters)
    """
    # Convert inputs to numpy arrays
    tokens_arr = np.asarray(tokens, dtype=float)
    model_arr = np.asarray(model_size, dtype=float)
    unique_arr = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Objective: mean squared error
    def objective(params):
        pred = scaling_law_func(tokens_arr, model_arr, unique_arr, params)
        return np.mean((pred - y)**2)

    # Bounds to enforce positivity where needed
    bounds = [
        (None, None),      # E: unbounded
        (1e-8, None),      # A > 0
        (1e-8, None),      # α > 0
        (1e-8, None),      # B > 0
        (1e-8, None),      # β > 0
        (1e-8, None),      # R_N > 0
        (1e-8, None)       # R_D > 0
    ]

    # Heuristic initial guess based on data range
    E0 = max(np.min(y) * 0.9, 1e-8)
    span = np.max(y) - np.min(y)
    A0 = max(span, 1e-8)
    B0 = max(span * 0.5, 1e-8)
    alpha0, beta0 = 0.5, 0.5
    R0 = 0.1
    base_init = np.array([E0, A0, alpha0, B0, beta0, R0, R0], dtype=float)

    best_params = None
    best_loss = np.inf

    # Multi-start optimization: base init + random perturbations
    inits = [base_init]
    for _ in range(4):
        jitter = 0.3 * np.random.randn(7)
        init = base_init * (1 + jitter)
        # enforce positivity for parameters 1..6
        init[1:] = np.maximum(init[1:], 1e-8)
        inits.append(init)

    for init in inits:
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception:
            continue

    # Fallback to base guess if optimization fails
    final_params = best_params if best_params is not None else base_init
    return final_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 7