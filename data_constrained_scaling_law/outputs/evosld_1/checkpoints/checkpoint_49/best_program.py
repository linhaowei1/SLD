# EVOLVE-BLOCK-START
"""
Refined data-constrained scaling law for LLM training loss.

We model the loss as a sum of two diminishing-return power-law terms plus a floor:
  1) Model-size term:       B * (model_size_GB)^(-alpha)
  2) Effective-data term:   C * (eff_data_GB)^(-beta)
  3) Floor (irreducible):   A

Where the effective data (in GB) saturates with repetition via a Hill-like form:
  eff_data_GB = u_GB * (1 – exp[ – (tokens_GB / (k * u_GB))^gamma ])

Total parameters: 7
  A     : irreducible loss floor
  B     : model-size amplitude
  C     : data-amplitude
  alpha : model-size exponent
  beta  : data exponent
  gamma : saturation sharpness
  k     : repetition scale factor
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss from:
      tokens         : array of training tokens used
      model_size     : array of model parameter counts
      unique_tokens  : array of available unique tokens
      params         : [A, B, C, alpha, beta, gamma, k]
    Returns predicted loss array.
    """
    A, B, C, alpha, beta, gamma, k = params

    # Normalize to billions for numerical stability
    ms = np.maximum(model_size / 1e9, 1e-6)       # model size in GB
    tk = np.maximum(tokens     / 1e9, 1e-6)       # training tokens in GB
    u  = np.maximum(unique_tokens / 1e9, 1e-6)    # unique tokens in GB

    # Hill-like saturation of effective data
    ratio = tk / (k * u)
    # avoid negative or zero inside power
    ratio = np.maximum(ratio, 1e-12)
    eff = u * (1.0 - np.exp(- ratio**gamma))
    eff = np.maximum(eff, 1e-12)

    # Compute two power-law terms plus floor
    term_size = B * ms**(-alpha)
    term_data = C * eff**(-beta)

    return A + term_size + term_data

# annotate number of parameters
scaling_law_func.num_params = 7

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law by minimizing MSE:
      1) Global search via differential evolution
      2) Local refinement via L-BFGS-B
    """
    # Parameter bounds
    bounds = [
        (0.0,   10.0),    # A: irreducible loss floor
        (1e-6, 100.0),    # B: model-size amplitude
        (1e-6, 100.0),    # C: data amplitude
        (1e-3,   5.0),    # alpha: model exponent
        (1e-3,   5.0),    # beta: data exponent
        (0.1,    5.0),    # gamma: saturation sharpness
        (1e-2, 100.0),    # k: repetition scale
    ]

    # Pre-normalize inputs once
    ms_arr = np.maximum(model_size / 1e9, 1e-6)
    tk_arr = np.maximum(tokens     / 1e9, 1e-6)
    u_arr  = np.maximum(unique_tokens / 1e9, 1e-6)

    def _mse(params):
        A, B, C, alpha, beta, gamma, k = params
        # effective data
        ratio = tk_arr / (k * u_arr)
        ratio = np.maximum(ratio, 1e-12)
        eff = u_arr * (1.0 - np.exp(- ratio**gamma))
        eff = np.maximum(eff, 1e-12)
        # predictions
        pred = A + B * ms_arr**(-alpha) + C * eff**(-beta)
        return np.mean((pred - loss_values)**2)

    # Global optimization
    result_de = differential_evolution(
        _mse,
        bounds,
        maxiter=200,
        popsize=20,
        tol=1e-6,
        polish=False,
        disp=False
    )
    x0 = result_de.x

    # Local refinement
    result_lb = minimize(
        _mse,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-9}
    )

    if result_lb.success:
        return result_lb.x
    else:
        return x0
# EVOLVE-BLOCK-END