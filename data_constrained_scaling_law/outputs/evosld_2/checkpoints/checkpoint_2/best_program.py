# EVOLVE-BLOCK-START
"""
Enhanced data-constrained scaling law discovery for LLM training scenarios.
We model two core contributions to loss:
  - model-size term:     a * model_size^{-alpha}
  - effective-data term: b * [u * (1 - exp(-tokens/(gamma * u)))]^{-beta}
plus a baseline loss floor L_inf.

Total parameters: 6 (<= 7).
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict loss given:
      tokens        : array of training tokens used
      model_size    : array of model parameter counts
      unique_tokens : array of unique tokens available
      params        : array([L_inf, a, alpha, b, gamma, beta])
    Returns:
      predicted loss array of same shape
    """
    # Unpack parameters
    L_inf, a, alpha, b, gamma, beta = params
    
    # Scale quantities to billions for numerical stability
    t = tokens        / 1e9  # training tokens in billions
    m = model_size    / 1e9  # model size in billions
    u = unique_tokens / 1e9  # unique tokens in billions
    
    # Model-size contribution: decays as a power law
    model_term = a * np.power(m + 1e-12, -alpha)
    
    # Effective data = u * (1 - exp(-t/(gamma*u)))
    # captures saturation when tokens >> unique_tokens
    eff_data = u * (1 - np.exp(-t / (gamma * u + 1e-12)))
    # Data contribution: decays as a power law of effective data
    data_term  = b * np.power(eff_data + 1e-12, -beta)
    
    # Total predicted loss
    return L_inf + model_term + data_term

# Number of parameters used
scaling_law_func.num_params = 6

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 6-parameter scaling law to observed loss data.
    Uses a robust Huber loss for stability against outliers.
    """
    # Initial guesses
    L_inf0  = np.min(loss_values)
    a0       = 1e3
    alpha0   = 0.5
    b0       = 1e3
    gamma0   = 1.0
    beta0    = 0.5
    x0 = np.array([L_inf0, a0, alpha0, b0, gamma0, beta0], dtype=float)
    
    # Bounds to keep parameters in reasonable ranges
    bounds = [
        (0.0,    10.0),    # L_inf
        (1e-6,   1e6),     # a
        (1e-3,   5.0),     # alpha
        (1e-6,   1e6),     # b
        (1e-2,   10.0),    # gamma
        (1e-3,   5.0)      # beta
    ]
    
    def huber_loss(residuals, delta=1.0):
        """Compute sum of Huber loss over residuals."""
        abs_r = np.abs(residuals)
        mask  = abs_r <= delta
        sq    = 0.5 * residuals[mask]**2
        lin   = delta * (abs_r[~mask] - 0.5 * delta)
        return np.sum(sq) + np.sum(lin)
    
    def objective(params):
        # if any param is NaN or outside bounds, penalize heavily
        if np.any(np.isnan(params)):
            return 1e9
        pred = scaling_law_func(tokens, model_size, unique_tokens, params)
        if np.any(~np.isfinite(pred)):
            return 1e9
        resid = pred - loss_values
        return huber_loss(resid, delta=1.0)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter':10000, 'ftol':1e-12}
    )
    
    if result.success:
        fitted = result.x
    else:
        # fallback to initial guess on failure
        fitted = x0
    
    return fitted
# EVOLVE-BLOCK-END