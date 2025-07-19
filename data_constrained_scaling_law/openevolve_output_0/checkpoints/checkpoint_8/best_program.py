# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Constrained‐data scaling law with saturation and interaction terms.
    
    params = [L_inf, a, c, alpha, beta, gamma, k]
      L_inf : irreducible loss floor
      a     : scale for model+data interaction term
      c     : scale for dataset diversity term
      alpha : exponent on model size
      beta  : exponent on effective tokens
      gamma : exponent on unique tokens
      k     : saturation rate for repeated data
    """
    L_inf, a, c, alpha, beta, gamma, k = params

    # Normalize to gigascale for numerical stability
    ms = model_size / 1e9
    u  = unique_tokens / 1e9

    # Effective token usage: saturates as tokens >> unique_tokens
    ratio = tokens / (unique_tokens + 1e-12)
    t_eff = unique_tokens * (1.0 - np.exp(-k * ratio))
    t_norm = t_eff / 1e9

    # Combined scaling-law form
    loss = (
        L_inf
        + a * np.power(ms, -alpha) * np.power(t_norm, -beta)
        + c * np.power(u, -gamma)
    )
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the above scaling law by optimizing a log‐parametrization
    to enforce positivity and improve conditioning.
    """
    # Map raw -> actual via exp to keep all params > 0
    def unpack(raw):
        return np.exp(raw)

    def objective(raw):
        params = unpack(raw)
        pred   = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values) ** 2)

    # Initial guesses for [L_inf,a,c,alpha,beta,gamma,k]
    init_actual = np.array([0.5, 1e-2, 1e-2, 0.8, 0.6, 0.4, 1.0])
    init_raw    = np.log(init_actual)

    res = minimize(
        objective,
        init_raw,
        method='L-BFGS-B',
        options={'maxiter': 10000, 'ftol': 1e-9}
    )

    raw_opt    = res.x if res.success else init_raw
    params_opt = unpack(raw_opt)

    # Record number of params for external checks
    scaling_law_func.num_params = len(params_opt)
    return params_opt

# Declare expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END