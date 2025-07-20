# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(num_experts, total_parameter_count, params):
    """
    A log–quadratic scaling law for MoE models:
      log(loss) = θ0 + θ1·log(E) + θ2·log(P)
                  + θ3·log(E)·log(P)
                  + θ4·[log(E)]^2 + θ5·[log(P)]^2
      loss = exp(log(loss))
    where:
      E = num_experts (clamped ≥1)
      P = total_parameter_count (clamped ≥1)
      params = [θ0, θ1, θ2, θ3, θ4, θ5]
    """
    E = np.maximum(np.asarray(num_experts, dtype=float), 1.0)
    P = np.maximum(np.asarray(total_parameter_count, dtype=float), 1.0)

    x1 = np.log(E)
    x2 = np.log(P)

    θ0, θ1, θ2, θ3, θ4, θ5 = params
    log_L = (θ0
             + θ1 * x1
             + θ2 * x2
             + θ3 * x1 * x2
             + θ4 * x1**2
             + θ5 * x2**2)

    return np.exp(log_L)


def fit_scaling_law(num_experts, total_parameter_count, loss_values):
    """
    Fit the log–quadratic scaling law via linear least squares
    on (log E, log P) features.
    Returns optimized params [θ0, θ1, θ2, θ3, θ4, θ5].
    """
    E = np.maximum(np.asarray(num_experts, dtype=float), 1.0)
    P = np.maximum(np.asarray(total_parameter_count, dtype=float), 1.0)
    L = np.maximum(np.asarray(loss_values, dtype=float), 1e-8)

    x1 = np.log(E)
    x2 = np.log(P)
    y = np.log(L)

    # Build design matrix [1, x1, x2, x1*x2, x1^2, x2^2]
    X = np.vstack([
        np.ones_like(x1),
        x1,
        x2,
        x1 * x2,
        x1**2,
        x2**2
    ]).T

    # Solve for params via least squares
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    return params

# Annotate number of parameters
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END