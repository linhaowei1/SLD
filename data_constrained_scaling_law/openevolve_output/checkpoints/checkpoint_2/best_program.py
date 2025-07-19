# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios
Evolved version: models loss as sum of parameter‐scaling, unique‐data‐scaling,
and effective‐token‐scaling terms with positivity and stability constraints.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given training tokens, model size, and available unique tokens,
    using a 7‐parameter form that accounts for data saturation.
    
    params: [L0, a, alpha, b, beta, c, gamma]
      L0    : irreducible loss floor
      a      : scale factor for model_size term
      alpha  : exponent on model_size
      b      : scale factor for unique_tokens term
      beta   : exponent on unique_tokens
      c      : scale factor for effective tokens term
      gamma  : exponent on effective tokens
    
    Effective tokens = U * (1 - exp(-T/U)), capturing diminishing returns
    when T >> U.
    """
    L0, a, alpha, b, beta, c, gamma = params
    # avoid division by zero
    U = unique_tokens.astype(np.float64)
    T = tokens.astype(np.float64)
    # ratio and effective token count
    ratio = T / (U + 1e-12)
    eff_T = U * (1.0 - np.exp(-ratio))
    # compute each term
    term_model = a * np.power(model_size + 1e-12, -alpha)
    term_unique = b * np.power(U + 1e-12,        -beta)
    term_eff    = c * np.power(eff_T + 1e-12,    -gamma)
    return L0 + term_model + term_unique + term_eff

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7‐parameter scaling law via unconstrained optimization in log‐space
    to enforce positivity of all parameters.
    Returns params = [L0, a, alpha, b, beta, c, gamma].
    """
    T = np.asarray(tokens,        dtype=np.float64)
    N = np.asarray(model_size,    dtype=np.float64)
    U = np.asarray(unique_tokens, dtype=np.float64)
    Y = np.asarray(loss_values,   dtype=np.float64)

    # Transform from unconstrained v to positive params via exp(v)
    def unpack(v):
        return np.exp(v)

    # Objective: MSE between predicted and actual loss
    def obj(v):
        p = unpack(v)
        pred = scaling_law_func(T, N, U, p)
        return np.mean((pred - Y)**2)

    # Good initial guesses (in log‐space)
    L0_init    = max(np.min(Y)*0.5, 1e-2)
    a_init     = 1.0
    alpha_init = 0.5
    b_init     = 1.0
    beta_init  = 0.5
    c_init     = 1.0
    gamma_init = 0.5
    v0 = np.log([L0_init, a_init, alpha_init, b_init, beta_init, c_init, gamma_init])

    # Optimize with L-BFGS-B
    res = minimize(
        obj,
        v0,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-9}
    )

    # On failure, fall back to initial
    v_opt = res.x if res.success else v0
    params = unpack(v_opt)
    return params

# Specify number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END