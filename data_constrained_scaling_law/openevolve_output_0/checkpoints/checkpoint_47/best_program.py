"""
Evolved data-constrained scaling law for LLM training scenarios.

We model loss as:
    loss = p0
         + p1 * model_size^(−a)
         + p2 * D_eff^(−b)

where the effective token count D_eff saturates exponentially:
    D_eff = unique_tokens * (1 − exp(−tokens / (unique_tokens * c)))

This 6-parameter form [p0, p1, a, p2, b, c] captures:
  • p0: irreducible loss floor
  • p1, a: model-size coefficient & exponent
  • p2, b: effective-data coefficient & exponent
  • c:     saturation scale for repeated data

We fit in log-space (unconstrained) via L-BFGS-B for stability.
"""

import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss via a simplified data-constrained scaling law:
      loss = p0
           + p1 * model_size^(−a)
           + p2 * D_eff^(−b)

    with D_eff = unique_tokens * (1 − exp(−tokens / (unique_tokens * c))).

    Args:
        tokens:        array_like, number of training tokens used
        model_size:    array_like, number of model parameters
        unique_tokens: array_like, number of unique tokens available
        params:        length-6 array [p0, p1, a, p2, b, c]

    Returns:
        numpy.ndarray of predicted loss values
    """
    p0, p1, a, p2, b, c = params
    T = np.array(tokens,      dtype=float)
    N = np.array(model_size,  dtype=float)
    U = np.array(unique_tokens,dtype=float)

    # Exponential saturation for effective data
    # Avoid division by zero with a tiny epsilon
    D_eff = U * (1.0 - np.exp(- T / (U * c + 1e-12)))
    D_eff = np.maximum(D_eff, 1e-12)

    # Additive power-law contributions
    loss = p0 + p1 * N**(-a) + p2 * D_eff**(-b)
    return loss

# inform how many parameters this form expects
scaling_law_func.num_params = 6

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 6-parameter scaling law to observed losses by minimizing MSE
    in log-parameter space to enforce positivity.

    Args:
        tokens:         array_like, training tokens used
        model_size:     array_like, model size (parameters)
        unique_tokens:  array_like, unique tokens available
        loss_values:    array_like, observed loss values

    Returns:
        params_opt: length-6 numpy array of fitted parameters
    """
    T = np.array(tokens,       dtype=float)
    N = np.array(model_size,   dtype=float)
    U = np.array(unique_tokens,dtype=float)
    L = np.array(loss_values,  dtype=float)

    # Reasonable initial guesses
    p0_init = max(np.min(L) * 0.9, 1e-3)
    p1_init = (np.max(L) - np.min(L)) * 0.5
    a_init  = 0.3
    p2_init = p1_init
    b_init  = 0.3
    c_init  = 1.0
    init_params = np.array([p0_init, p1_init, a_init, p2_init, b_init, c_init])

    # Work in log-space for positivity
    theta0 = np.log(init_params + 1e-12)

    def mse_obj(theta):
        params = np.exp(theta)
        pred = scaling_law_func(T, N, U, params)
        return np.mean((pred - L)**2)

    res = minimize(
        mse_obj,
        theta0,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    if res.success:
        return np.exp(res.x)
    else:
        # Fallback to initial guess if optimization fails
        return init_params
# EVOLVE-BLOCK-END