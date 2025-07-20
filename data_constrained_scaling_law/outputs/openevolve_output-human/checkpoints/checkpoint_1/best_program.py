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
    Fit the human-designed data-constrained scaling law to data using L-BFGS-B with bounds and random restarts.
    
    Args:
        tokens: Array of training tokens used
        model_size: Array of model parameter counts
        unique_tokens: Array of unique tokens available
        loss_values: Array of corresponding loss values
        
    Returns:
        Optimized parameters (7 parameters)
    """
    # Ensure inputs are numpy arrays
    tokens = np.asarray(tokens, dtype=float)
    model_size = np.asarray(model_size, dtype=float)
    unique_tokens = np.asarray(unique_tokens, dtype=float)
    loss_values = np.asarray(loss_values, dtype=float)

    # Initial parameter estimates based on data ranges
    E0 = np.min(loss_values)
    A0 = (np.max(loss_values) - np.min(loss_values)) * 0.5
    B0 = A0
    alpha0 = 0.7
    beta0 = 0.7
    R_N0 = 1.0
    R_D0 = 1.0
    initial_params = np.array([E0, A0, alpha0, B0, beta0, R_N0, R_D0])

    # Define bounds to enforce positivity and reasonable parameter ranges
    bounds = [
        (0.0, np.max(loss_values) * 2.0),   # E
        (1e-8, np.max(loss_values) * 10.0), # A
        (1e-8, 5.0),                        # alpha
        (1e-8, np.max(loss_values) * 10.0), # B
        (1e-8, 5.0),                        # beta
        (1e-8, 10.0),                       # R_N
        (1e-8, 10.0)                        # R_D
    ]

    def objective(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values) ** 2)

    # Perform a few random-restart L-BFGS-B optimizations to avoid local minima
    best_result = None
    for restart in range(5):
        if restart == 0:
            x0 = initial_params.copy()
        else:
            # small random perturbation around the initial guess
            perturb = 1.0 + 0.3 * (np.random.rand(7) - 0.5)
            x0 = initial_params * perturb

        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxfun': 5000, 'ftol': 1e-9}
        )

        if res.success:
            if best_result is None or res.fun < best_result.fun:
                best_result = res

    # Fallback to the last result if none succeeded
    final_params = best_result.x if best_result is not None else res.x
    return final_params
# EVOLVE-BLOCK-END

# Set the number of parameters this function expects
scaling_law_func.num_params = 7