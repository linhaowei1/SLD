# EVOLVE-BLOCK-START
"""
MoE scaling law discovery for Mixture of Experts models.

Improved 6‐parameter form capturing expert scaling, parameter scaling,
and their interaction, fitted via bounded L-BFGS-B optimization for
numerical stability and interpretability.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A 6-parameter MoE scaling law:

      loss = a
           + b * (num_experts + 1)^(-c)
           + d * ((total_parameter_count + 1)/1e9)^(-e)
           + f * (num_experts + 1)^(-c) * ((total_parameter_count + 1)/1e9)^(-e)

    where:
      a  : asymptotic floor loss
      b  : expert-only coefficient
      c  : expert exponent
      d  : parameter-only coefficient
      e  : parameter exponent
      f  : interaction coefficient

    Args:
        num_experts (array_like): number of experts per model
        total_parameter_count (array_like): dense parameter counts per model
        params (array_like of length 6): [a, b, c, d, e, f]

    Returns:
        numpy.ndarray: predicted loss values
    """
    a, b, c, d, e, f = params
    # shift to avoid zero, scale parameters for numerical stability
    ne = np.asarray(num_experts, dtype=float) + 1.0
    tp = (np.asarray(total_parameter_count, dtype=float) + 1.0) / 1e9

    # compute power-law terms
    term_experts    = b * np.power(ne, -np.abs(c))
    term_params     = d * np.power(tp, -np.abs(e))
    term_interaction= f * np.power(ne, -np.abs(c)) * np.power(tp, -np.abs(e))

    return a + term_experts + term_params + term_interaction

def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the 6-parameter MoE scaling law to data via L-BFGS-B.

    Args:
        num_experts (array_like): number of experts
        total_parameter_count (array_like): dense parameter counts
        loss_values (array_like): observed validation losses

    Returns:
        numpy.ndarray: optimized [a, b, c, d, e, f]
    """
    num_experts = np.asarray(num_experts, dtype=float)
    total_parameter_count = np.asarray(total_parameter_count, dtype=float)
    loss_values = np.asarray(loss_values, dtype=float)

    # Initial guess: floor ~ median loss, coefficients ~ 1, exponents ~ 1
    a0 = np.median(loss_values)
    initial = np.array([a0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

    # Bounds: a unrestricted, b,d,f >=0, c,e in [1e-6, 10] for stability
    bounds = [
        (None, None),      # a
        (0.0, None),       # b
        (1e-6, 10.0),      # c
        (0.0, None),       # d
        (1e-6, 10.0),      # e
        (0.0, None)        # f
    ]

    def objective(params):
        pred = scaling_law_func(num_experts, total_parameter_count, params)
        # mean squared error
        return np.mean((pred - loss_values) ** 2)

    result = minimize(
        objective,
        initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 5000}
    )

    if not result.success:
        # fallback to initial if optimization fails
        return initial
    return result.x

# specify how many params the scaling law uses
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END