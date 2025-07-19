"""
Evolved data-constrained scaling law model for LLM training scenarios.

We model loss as a baseline floor plus two power‐law contributions:
  1) capacity term combining model size and effective data:
       p1 * M^(-α) * D_eff^(-β)
     where D_eff = unique_tokens * (1 - exp(-d * (tokens/unique_tokens)))
  2) (optional) a small-gain repetition penalty is implicitly handled
     by D_eff saturation.

Total parameters: 6
  p0    : irreducible loss floor
  p1    : scale for capacity term
  α     : exponent on model size
  p2    : scale for data term
  β     : exponent on effective data
  d     : data‐saturation rate

This form captures the interaction between model size and data under
strong data‐constrained regimes with limited unique tokens.
"""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given:
      tokens        : training tokens used (array-like)
      model_size    : model parameter counts (array-like)
      unique_tokens : unique tokens available (array-like)
      params        : [p0, p1, alpha, p2, beta, d]
    Returns:
      loss predictions (numpy array)
    """
    # unpack parameters
    p0, p1, alpha, p2, beta, d = params

    # ensure float arrays
    tk = np.asarray(tokens, dtype=float)
    ms = np.asarray(model_size, dtype=float)
    uq = np.asarray(unique_tokens, dtype=float)

    # small epsilon for numeric stability
    eps = 1e-12

    # compute effective data D_eff = uq * (1 - exp(-d * (tk/uq)))
    cov = tk / (uq + eps)
    D_eff = uq * (1.0 - np.exp(-d * cov))

    # compute terms
    term_M = p1 * np.power(ms + eps, -alpha)
    term_D = p2 * np.power(D_eff + eps, -beta)

    # total loss
    return p0 + term_M + term_D

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law parameters to observed losses.
    Uses multi-start L-BFGS-B to minimize mean squared error.
    Returns:
      optimized params [p0, p1, alpha, p2, beta, d]
    """
    # convert inputs to numpy arrays
    tk = np.asarray(tokens, dtype=float)
    ms = np.asarray(model_size, dtype=float)
    uq = np.asarray(unique_tokens, dtype=float)
    lv = np.asarray(loss_values, dtype=float)

    # objective: mean squared error
    def _mse(params):
        pred = scaling_law_func(tk, ms, uq, params)
        return np.mean((pred - lv) ** 2)

    # initial guess
    p0_init = max(0.0, np.min(lv) * 0.5)
    p1_init = 1e3
    alpha_init = 0.5
    p2_init = 1e3
    beta_init = 0.5
    d_init = 1.0
    init = np.array([p0_init, p1_init, alpha_init, p2_init, beta_init, d_init], dtype=float)

    # parameter bounds
    bounds = [
        (0.0, np.max(lv)),    # p0
        (1e-8, 1e6),          # p1
        (1e-3, 5.0),          # alpha
        (1e-8, 1e6),          # p2
        (1e-3, 5.0),          # beta
        (1e-6, 1e3)           # d
    ]

    best_params = None
    best_loss = np.inf

    # random generator for jitter
    rng = np.random.default_rng(123)

    # multi-start scales
    for scale in (0.5, 1.0, 2.0):
        trial = init * scale
        # small jitter
        trial = trial * (1.0 + 0.1 * (rng.random(trial.shape) - 0.5))
        res = minimize(
            _mse,
            trial,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # fallback
    if best_params is None:
        best_params = init

    return best_params

# metadata
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END