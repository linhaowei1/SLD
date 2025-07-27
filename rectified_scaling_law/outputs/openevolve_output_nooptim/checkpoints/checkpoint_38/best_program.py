# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios

We model the loss as a 4-parameter saturating power-law (Hill equation):
    loss(x) = d + a / (1 + (x / c)**b)

where:
    a > 0  controls the initial amplitude of decay,
    b > 0  governs the steepness of the decay (power exponent),
    c > 0  is a characteristic data-size scale,
    d > 0  is the asymptotic loss floor as x → ∞.

To enforce positivity of all four parameters, we optimize their logarithms
and then exponentiate inside the function.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A 4-parameter Hill-type (saturating power-law) scaling function.

    Args:
        data_points: array-like of training set sizes (e.g., [200, 1000, ...])
        params: array-like of 4 raw parameters [ln a, ln b, ln c, ln d]

    Returns:
        predicted loss values as a numpy array
    """
    x = np.asarray(data_points, dtype=float)
    # exponentiate to enforce positivity
    a = np.exp(params[0])
    b = np.exp(params[1])
    c = np.exp(params[2])
    d = np.exp(params[3])
    # compute the saturating power-law (Hill equation)
    return d + a / (1.0 + np.power(x / c, b))

def fit_scaling_law(data_points, loss_values):
    """
    Fit the scaling law to observed losses using BFGS to minimize MSE.

    Args:
        data_points: array-like of training set sizes
        loss_values: array-like of observed losses

    Returns:
        optimized raw parameters [ln a, ln b, ln c, ln d]
        (to get actual a,b,c,d, exponentiate these)
    """
    # initialize all four raw parameters to zero (so exp(0)=1)
    initial_params = np.zeros(4)

    def objective(params):
        try:
            preds = scaling_law_func(data_points, params)
            return np.mean((preds - loss_values) ** 2)
        except Exception:
            # penalize any invalid parameter region heavily
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# indicate how many raw parameters the scaling law expects
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END