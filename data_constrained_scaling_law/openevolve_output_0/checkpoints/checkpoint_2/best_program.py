# EVOLVE-BLOCK-START
"""
Evolved data-constrained scaling law model for LLM training scenarios.

We model loss as a sum of power-law contributions from model size,
total tokens, and data repetition (tokens/unique_tokens ratio), plus a
baseline floor. 7 parameters in total.

loss = p0
     + p1 * (model_size + 1)^(-p2)
     + p3 * (tokens + 1)^(-p4)
     + p5 * ( (tokens+1)/(unique_tokens+1) )^(-p6)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and unique tokens,
    using a 7-parameter power-law model.

    params = [p0, p1, alpha, p3, beta, p5, gamma]
    loss = p0
         + p1 * (model_size+1)^(-alpha)
         + p3 * (tokens+1)^(-beta)
         + p5 * ((tokens+1)/(unique_tokens+1))^(-gamma)
    """
    p0, p1, alpha, p3, beta, p5, gamma = params
    # Add small +1 offsets for numerical stability
    ms = model_size.astype(float) + 1.0
    tk = tokens.astype(float) + 1.0
    uq = unique_tokens.astype(float) + 1.0
    rep = tk / uq
    # compute power-law terms
    term_ms = p1 * np.power(ms, -alpha)
    term_tk = p3 * np.power(tk, -beta)
    term_rep = p5 * np.power(rep, -gamma)
    return p0 + term_ms + term_tk + term_rep

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law to observed loss data.
    Uses multiple starts and bounded L-BFGS-B optimization for robustness.
    """
    # Ensure numpy arrays
    tokens = np.asarray(tokens, dtype=float)
    model_size = np.asarray(model_size, dtype=float)
    unique_tokens = np.asarray(unique_tokens, dtype=float)
    loss_values = np.asarray(loss_values, dtype=float)

    # Objective: mean squared error
    def objective(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values) ** 2)

    # Initial guess: baseline ~ mean loss, coefficients small, exponents ~1
    init = np.array([
        np.clip(np.mean(loss_values), 0.0, None),  # p0
        1e6,   # p1 (scale for model_size term)
        0.5,   # alpha
        1e-2,  # p3 (scale for tokens term)
        0.5,   # beta
        1e-2,  # p5 (scale for repetition term)
        0.5    # gamma
    ], dtype=float)

    # Bounds: all parameters >= 0, exponents reasonable
    bounds = [
        (0.0, None),    # p0
        (1e-8, None),   # p1
        (1e-3, 5.0),    # alpha
        (1e-8, None),   # p3
        (1e-3, 5.0),    # beta
        (1e-8, None),   # p5
        (1e-3, 5.0)     # gamma
    ]

    best_params = None
    best_loss = np.inf

    # Multi-start optimization
    rng = np.random.default_rng(42)
    for scale in [1.0, 0.5, 2.0]:
        trial_init = init * scale
        # small random jitter around init
        trial_init = trial_init * (1.0 + 0.1 * (rng.random(init.shape) - 0.5))
        res = minimize(
            objective,
            trial_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # Fallback to initial guess if optimization failed
    if best_params is None:
        best_params = init

    return best_params

# Attach num_params metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END