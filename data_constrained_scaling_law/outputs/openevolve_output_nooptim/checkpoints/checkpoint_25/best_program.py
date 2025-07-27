# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios

We propose a smoother, more interpretable saturating form by using
generalized power‐means to model effective scales.  This enforces
that when training tokens far exceed model size, the effective token
scale caps at the model size, and likewise the effective model scale
caps at the unique token budget.

Scaling law (7 parameters):
  L(N, D, U) = E
             + A / [eff_N(N, D)]^α
             + B / [eff_D(D, U)]^β

where
  eff_N(N, D) = (N^(−R_N) + D^(−R_N))^(−1/R_N)
  eff_D(D, U) = (D^(−R_D) + U^(−R_D))^(−1/R_D)

Parameters are constrained via absolute‐value transforms to ensure
positivity where needed, while E remains unconstrained.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Compute predicted loss L(N,D,U) given raw parameters.

    Args:
        tokens       : array-like of training tokens (N)
        model_size   : array-like of model parameter counts (D)
        unique_tokens: array-like of unique tokens available (U)
        params       : iterable of 7 parameters [E, A, alpha, B, beta, R_N, R_D]

    Returns:
        np.ndarray of predicted losses.
    """
    # unpack
    E, A, alpha, B, beta, R_N, R_D = params

    # enforce positivity where required
    A = np.abs(A)
    B = np.abs(B)
    α = np.abs(alpha) + 1e-6
    β = np.abs(beta)  + 1e-6
    rN = np.abs(R_N)  + 1e-6
    rD = np.abs(R_D)  + 1e-6

    # arrayify
    N = np.asarray(tokens,       dtype=float)
    D = np.asarray(model_size,   dtype=float)
    U = np.asarray(unique_tokens,dtype=float)

    # effective token scale: saturates at model size D
    eff_N = (N**(-rN) + D**(-rN))**(-1.0 / rN)

    # effective model scale: saturates at unique token budget U
    eff_D = (D**(-rD) + U**(-rD))**(-1.0 / rD)

    # loss prediction
    return E + A * eff_N**(-α) + B * eff_D**(-β)


def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law to data points using BFGS from ones initialization.
    Returns the 7 optimized parameters.
    """
    initial_params = np.ones(7)

    def objective(p):
        try:
            pred = scaling_law_func(tokens, model_size, unique_tokens, p)
            return np.mean((pred - loss_values)**2)
        except:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# for API compatibility
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END