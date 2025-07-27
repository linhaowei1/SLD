# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data‐constrained scaling law with an effective‐data term.
    
    D_eff = U * (1 - exp(-D/U))
    L(N, D, U) = L_inf
                  + a * N^{-alpha}
                  + b * D_eff^{-beta}
                  + c * (N * D_eff)^{-gamma}

    params = [L_inf, a, alpha, b, beta, c, gamma]  (7 parameters)
    """
    L_inf, a, alpha, b, beta, c, gamma = params
    # Compute effective unique tokens seen, saturating at U
    D_eff = unique_tokens * (1 - np.exp(-tokens / (unique_tokens + 1e-12)))
    # Prevent zero‐division or negative exponents
    N_term  = np.power(model_size + 1e-12, -alpha)
    D_term  = np.power(D_eff      + 1e-12, -beta)
    ND_term = np.power(model_size * D_eff + 1e-12, -gamma)
    loss = L_inf + a * N_term + b * D_term + c * ND_term
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7 parameters of the scaling law to (tokens, model_size, unique_tokens) → loss.
    Uses MSE objective and L‐BFGS‐B with simple bounds for positivity and stability.
    """
    # Convert inputs to numpy arrays
    tokens       = np.asarray(tokens, dtype=float)
    model_size   = np.asarray(model_size, dtype=float)
    unique_tokens= np.asarray(unique_tokens, dtype=float)
    loss_values  = np.asarray(loss_values, dtype=float)

    # Initialize parameters based on simple heuristics
    L_inf0   = max(1e-3, loss_values.min() * 0.9)
    residual = max(1e-3, loss_values.mean() - L_inf0)
    a0, b0, c0 = residual * 0.3, residual * 0.3, residual * 0.4
    alpha0, beta0, gamma0 = 0.3, 0.2, 0.1

    x0 = np.array([L_inf0, a0, alpha0, b0, beta0, c0, gamma0])

    # Bounds: L_inf, a, b, c >= 0; exponents >= 1e-6 and <= 5
    bounds = [
        (0,   None),  # L_inf
        (0,   None),  # a
        (1e-6, 5),    # alpha
        (0,   None),  # b
        (1e-6, 5),    # beta
        (0,   None),  # c
        (1e-6, 5)     # gamma
    ]

    def objective(params):
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values) ** 2)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 2000}
    )

    if result.success:
        return result.x
    else:
        # Fallback to initial guess on failure
        return x0

# Attach metadata
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END