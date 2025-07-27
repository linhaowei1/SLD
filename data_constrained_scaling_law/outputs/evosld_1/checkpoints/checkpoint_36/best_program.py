# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Data‐constrained scaling law with a saturating effective‐data term.

    loss ≈ A
         + B_m * (model_size/1e6)^(-α)
         + B_d * (eff_data)^(-β)

    where:
      cov_norm      = tokens / (unique_tokens + ε)
      unique_norm   = unique_tokens / 1e9
      eff_data      = unique_norm * (1 - exp(-cov_norm / φ))
    
    params:
      params[0] = A            # irreducible loss floor
      params[1] = log(B_m)     # model‐size prefactor
      params[2] = log(α)       # model‐size exponent
      params[3] = log(B_d)     # effective‐data prefactor
      params[4] = log(β)       # effective‐data exponent
      params[5] = log(φ)       # saturation scale for data coverage
    """
    A    = params[0]
    B_m  = np.exp(params[1])
    α    = np.exp(params[2])
    B_d  = np.exp(params[3])
    β    = np.exp(params[4])
    φ    = np.exp(params[5])

    # normalize for stability
    m_norm      = model_size / 1e6 + 1e-12
    unique_norm = unique_tokens / 1e9 + 1e-12
    cov_norm    = tokens / (unique_tokens + 1e-12)

    # saturating effective‐data term
    eff_data = unique_norm * (1.0 - np.exp(-cov_norm / φ))
    eff_data = np.maximum(eff_data, 1e-12)

    # compute loss
    loss = (
        A
        + B_m * np.power(m_norm, -α)
        + B_d * np.power(eff_data, -β)
    )
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit up to 6 parameters by minimizing mean squared error.
    Returns optimized [A, log B_m, log α, log B_d, log β, log φ].
    """
    tokens       = np.array(tokens,       dtype=float)
    model_size   = np.array(model_size,   dtype=float)
    unique_tokens= np.array(unique_tokens,dtype=float)
    loss_values  = np.array(loss_values,  dtype=float)

    # initial guesses
    A0 = max(np.min(loss_values) * 0.8, 1e-3)
    p0 = np.array([
        A0,              # A
        np.log(0.5),     # log B_m
        np.log(0.5),     # log α
        np.log(0.5),     # log B_d
        np.log(0.5),     # log β
        np.log(1.0)      # log φ
    ], dtype=float)

    # bounds: A ≥ 0, φ ≥ 1e-3 (in log‐space)
    bounds = [
        (0.0,       None),   # A
        (None,      None),   # log B_m
        (None,      None),   # log α
        (None,      None),   # log B_d
        (None,      None),   # log β
        (np.log(1e-3), None) # log φ
    ]

    def mse_obj(p):
        pred = scaling_law_func(tokens, model_size, unique_tokens, p)
        return np.mean((pred - loss_values)**2)

    res = minimize(
        mse_obj, p0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol':1e-12, 'gtol':1e-8, 'maxiter':5000}
    )

    return res.x if res.success else p0

# communicate expected param count
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END