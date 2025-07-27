# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios
Evolved version: incorporates model-data interplay and duplication effects,
uses parameter bounds and multiple restarts for robust fitting.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data-constrained scaling law function:
      loss = p0 
           + p1 * (model_size_scaled)^(-p2)
           + p3 * (tokens_scaled)^(-p4)
           + p5 * (tokens_scaled/unique_scaled)^(-p6)

    Args:
        tokens: Array of training tokens used (float)
        model_size: Array of model parameter counts (float)
        unique_tokens: Array of unique tokens available (float)
        params: Array of 7 parameters [p0,p1,p2,p3,p4,p5,p6]

    Returns:
        Array of predicted loss values
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6 = params

    # Scale quantities to ~O(1) for numerical stability
    M = model_size / 1e9 + 1e-12
    T = tokens / 1e9 + 1e-12
    U = unique_tokens / 1e9 + 1e-12
    dup = T / U

    # Compute loss
    loss = (
        p0
        + p1 * M**(-p2)
        + p3 * T**(-p4)
        + p5 * dup**(-p6)
    )
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the evolved scaling law to observed loss data.

    Args:
        tokens: Array of training tokens used
        model_size: Array of model parameter counts
        unique_tokens: Array of unique tokens available
        loss_values: Array of observed loss values

    Returns:
        Array of 7 optimized parameters [p0...p6]
    """
    # Convert inputs to numpy arrays
    tokens = np.asarray(tokens, dtype=float)
    model_size = np.asarray(model_size, dtype=float)
    unique_tokens = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Parameter bounds: ensure physical/empirical constraints
    y_max = np.max(y)
    bounds = [
        (0.0,             y_max * 2.0),   # p0: baseline loss offset
        (1e-8,            y_max * 10.0),  # p1: model-size scale factor
        (1e-3,            10.0),          # p2: model-size exponent
        (1e-8,            y_max * 10.0),  # p3: data-size scale factor
        (1e-3,            10.0),          # p4: data-size exponent
        (1e-8,            y_max * 10.0),  # p5: duplication scale factor
        (1e-3,            10.0)           # p6: duplication exponent
    ]

    # Objective: mean squared error between predicted and true loss
    def objective(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - y)**2)

    # Multi-start to avoid local minima
    best_params = None
    best_obj = np.inf
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Random initial guess within bounds
        init = np.array([rng.uniform(low, high) for (low, high) in bounds])
        res = minimize(
            objective,
            init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x

    # Fallback to midpoint if all restarts fail
    if best_params is None:
        best_params = np.array([(low + high) / 2.0 for (low, high) in bounds])

    return best_params

# Attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END