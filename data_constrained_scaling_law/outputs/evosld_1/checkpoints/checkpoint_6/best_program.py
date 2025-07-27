# EVOLVE-BLOCK-START
"""
Refined data-constrained scaling law discovery for LLM training scenarios.

Key improvements:
1. Simplified three-term form capturing model-only, data-only, and model–data interplay.
2. Automatic median-based normalization for numerical stability.
3. Two-stage global + local optimization for robust fitting.
4. Clear, concise, and well-documented code.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss from training tokens, model size, and unique-token constraint.

    loss = p0
         + p1 * M^{-p2}
         + p3 * T^{-p4}
         + p5 * (M * T / U)^{-p6}

    where
      M = model_size / median_model_size
      T = tokens / median_tokens
      U = unique_tokens / median_unique_tokens

    Args:
        tokens: Array of training tokens used
        model_size: Array of model parameter counts
        unique_tokens: Array of unique tokens available
        params: Array of 7 parameters [p0,p1,p2,p3,p4,p5,p6]

    Returns:
        Array of predicted loss values
    """
    p0, p1, p2, p3, p4, p5, p6 = params

    # Prevent zero-divisions
    eps = 1e-12
    M = model_size / np.median(model_size) + eps
    T = tokens / np.median(tokens) + eps
    U = unique_tokens / np.median(unique_tokens) + eps

    # Core terms
    term_model     = p1 * M ** (-p2)
    term_data      = p3 * T ** (-p4)
    term_interplay = p5 * (M * T / U) ** (-p6)

    return p0 + term_model + term_data + term_interplay

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the refined scaling law to observed loss data via
    a two-stage global + local optimization.

    Returns optimized parameters [p0..p6].
    """
    # Convert to numpy
    T = np.asarray(tokens, dtype=float)
    Msz = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Upper scale for bounds
    y_max = np.max(y)
    # (low, high) for each param
    bounds = [
        (0.0,      y_max * 2.0),   # p0: baseline offset
        (1e-8,     y_max * 10.0),  # p1: model-scale
        (1e-4,     5.0),           # p2: model-exponent
        (1e-8,     y_max * 10.0),  # p3: data-scale
        (1e-4,     5.0),           # p4: data-exponent
        (1e-8,     y_max * 10.0),  # p5: interplay-scale
        (1e-4,     5.0)            # p6: interplay-exponent
    ]

    # Objective: mean squared error
    def mse_obj(p):
        pred = scaling_law_func(T, Msz, U, p)
        return np.mean((pred - y) ** 2)

    # Stage 1: global search via differential evolution
    result_de = differential_evolution(
        mse_obj,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=False,
        seed=0
    )
    p0 = result_de.x

    # Stage 2: local refinement via L-BFGS-B
    result_lb = minimize(
        mse_obj,
        p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    # Choose the best
    if result_lb.success and result_lb.fun <= result_de.fun:
        best_params = result_lb.x
    else:
        best_params = result_de.x

    return best_params

# Attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END