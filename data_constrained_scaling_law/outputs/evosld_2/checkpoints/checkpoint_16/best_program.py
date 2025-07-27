# EVOLVE-BLOCK-START
"""
Improved data‐constrained scaling law for LLM training loss modeling.

We model loss as a sum of:
  1) irreducible floor:                L_inf
  2) model‐size power‐law term:         A * (M_norm)^(-alpha)
  3) data‐coverage power‐law term:      B * (D_norm)^(-beta)
  4) interaction term capturing synergy: C * (M_norm)^(-alpha) * (D_norm)^(-beta)

where
  M_norm = model_size / 1e9
  D_norm = U_norm * (1 - exp(-T / (phi * U + eps)))
  U_norm = unique_tokens / 1e9

Total parameters (7):
  params = [L_inf, A, alpha, B, beta, phi, C]
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and unique tokens.
    Args:
        tokens       : array-like, number of training tokens used (T)
        model_size   : array-like, model parameter count (M)
        unique_tokens: array-like, unique tokens available (U)
        params       : length-7 array [L_inf, A, alpha, B, beta, phi, C]
    Returns:
        loss_pred: array of predicted loss values
    """
    L_inf, A, alpha, B, beta, phi, C = params
    T = np.asarray(tokens, dtype=np.float64)
    M = np.asarray(model_size, dtype=np.float64)
    U = np.asarray(unique_tokens, dtype=np.float64)
    # Normalize scales to billions to stabilize exponents
    M_norm = M / 1e9
    U_norm = U / 1e9
    eps = 1e-12
    # Effective data coverage (normalized)
    D_norm = U_norm * (1 - np.exp(-T / (phi * U + eps)))
    # Power-law contributions
    m_term = np.power(M_norm + eps, -alpha)
    d_term = np.power(D_norm + eps, -beta)
    # Combined loss
    return L_inf + A * m_term + B * d_term + C * m_term * d_term

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7‐parameter scaling law to observed losses via MSE minimization.
    Args:
        tokens       : array of training tokens used
        model_size   : array of model parameter counts
        unique_tokens: array of unique tokens available
        loss_values  : array of observed loss values
    Returns:
        best_params: length-7 array of optimized parameters
    """
    T = np.asarray(tokens, dtype=np.float64)
    M = np.asarray(model_size, dtype=np.float64)
    U = np.asarray(unique_tokens, dtype=np.float64)
    L = np.asarray(loss_values, dtype=np.float64)
    # Initial parameter guess
    L0 = max(1e-3, np.min(L) * 0.8)
    range_L = max(1e-3, np.max(L) - np.min(L))
    x0 = np.array([
        L0,           # L_inf
        range_L,      # A
        0.5,          # alpha
        range_L,      # B
        0.5,          # beta
        1.0,          # phi
        range_L * 0.5 # C
    ], dtype=np.float64)
    # Bounds to ensure positivity and reasonable exponents
    bnds = [
        (0.0, np.max(L)),    # L_inf
        (1e-8, None),        # A
        (1e-6, 10.0),        # alpha
        (1e-8, None),        # B
        (1e-6, 10.0),        # beta
        (1e-6, 10.0),        # phi
        (0.0, None)          # C
    ]
    # Objective: mean squared error
    def objective(p):
        pred = scaling_law_func(T, M, U, p)
        return np.mean((pred - L) ** 2)
    res = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )
    return res.x if res.success else x0

# annotate number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END