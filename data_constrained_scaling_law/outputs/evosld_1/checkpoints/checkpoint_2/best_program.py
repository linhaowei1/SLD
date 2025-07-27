# EVOLVE-BLOCK-START
"""
Evolved data-constrained scaling law discovery for LLM training scenarios.
Model: Loss = A + B * (model_size)^(-α) + C * (tokens)^(-β) + D * (tokens/unique_tokens)^(-γ)
7 parameters: [A, B, α, C, β, D, γ]
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and unique tokens.

    params = [A, B, alpha, C, beta, D, gamma]
      A     : baseline loss offset
      B, C, D : amplitude coefficients
      alpha, beta, gamma : positive exponents

    Loss = A + B*ms**(-alpha) + C*tk**(-beta) + D*ur**(-gamma)
      where ms = model_size / 1e9, tk = tokens / 1e9, ur = min(tk/ut, 1.0)
    """
    # unpack
    A, B, alpha, C, beta, D, gamma = params
    # normalize inputs for stability
    ms = model_size / 1e9 + 1e-6
    tk = tokens / 1e9 + 1e-6
    ut = unique_tokens / 1e9 + 1e-6
    ur = np.minimum(tk / ut, 1.0)
    # power-law components
    loss = (
        A
        + B * np.power(ms, -alpha)
        + C * np.power(tk, -beta)
        + D * np.power(ur, -gamma)
    )
    return loss

# number of parameters for bookkeeping
scaling_law_func.num_params = 7

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law to empirical loss data.
    Uses L-BFGS-B with simple bounds and multiple restarts.
    """
    # bounds: A∈[0,10], B,C,D∈[0,10], exponents ∈[1e-3, 2]
    bounds = [
        (0.0, 10.0),   # A
        (0.0, 10.0),   # B
        (1e-3, 2.0),   # alpha
        (0.0, 10.0),   # C
        (1e-3, 2.0),   # beta
        (0.0, 10.0),   # D
        (1e-3, 2.0),   # gamma
    ]

    # objective: mean squared error
    def _mse(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values) ** 2)

    best_params = None
    best_loss = np.inf

    # deterministic initial guesses + small random perturbations
    base_guess = np.array([np.mean(loss_values), 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
    for i in range(5):
        init = base_guess * (1.0 + 0.1 * (np.random.rand(7) - 0.5))
        res = minimize(
            _mse,
            init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # fallback to base if optimization failed
    if best_params is None:
        best_params = base_guess

    return best_params
# EVOLVE-BLOCK-END