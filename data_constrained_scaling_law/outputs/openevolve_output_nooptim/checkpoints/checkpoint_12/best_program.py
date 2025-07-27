# EVOLVE-BLOCK-START
"""
Data-constrained scaling law discovery for LLM training scenarios

Simplified, saturating two‐term power‐law model to improve fit stability
and capture both data‐limited and model‐limited regimes:

  L(N,D,U) = E
            + A / (vis_N^alpha)
            + B / (vis_D^beta)

where saturation scales are

  vis_N = U * N / (N + R_N * U)
  vis_D = U * D / (D + R_D * U)

Params:
  0: E      – irreducible error floor
  1: A      – coefficient for data term
  2: alpha  – exponent for data term
  3: B      – coefficient for model term
  4: beta   – exponent for model term
  5: R_N    – data‐saturation scale (relative to U)
  6: R_D    – model‐saturation scale (relative to U)
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    E, A, alpha, B, beta, R_N, R_D = params
    eps = 1e-12

    # ensure no zero divisions
    U = unique_tokens + eps
    N = tokens
    D = model_size

    # saturating effective data and model scales
    vis_N = U * N / (N + R_N * U + eps)
    vis_D = U * D / (D + R_D * U + eps)

    # two‐term power‐law loss
    loss = (
        E
        + A / (np.power(vis_N, alpha) + eps)
        + B / (np.power(vis_D, beta) + eps)
    )
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law to observed (tokens, model_size, unique_tokens) → loss_values

    Uses BFGS starting from all‐ones initialization.
    """
    initial_params = np.ones(7)

    def objective(params):
        try:
            pred = scaling_law_func(tokens, model_size, unique_tokens, params)
            return np.mean((pred - loss_values) ** 2)
        except:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# declare expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END