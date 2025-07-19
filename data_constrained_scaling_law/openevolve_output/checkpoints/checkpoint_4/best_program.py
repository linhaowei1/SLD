# EVOLVE-BLOCK-START
"""
Evolved data-constrained scaling law for LLM training:
Models loss as a sum of a model-size term and an effective-data term
that saturates when data repetition increases.
Uses at most 7 parameters (here 6).
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predicts loss given:
      tokens        : array of training tokens used
      model_size    : array of model parameter counts
      unique_tokens : array of unique tokens available
      params        : [a0, C_N, alpha, C_D, beta, gamma]
                       a0     : baseline loss
                       C_N    : coefficient for model-size term
                       alpha  : exponent for model-size
                       C_D    : coefficient for data term
                       beta   : exponent for effective-data
                       gamma  : saturation rate for data repetition

    Returns:
      loss predictions (same shape as inputs)
    """
    a0, C_N, alpha, C_D, beta, gamma = params
    # Enforce positivity for coefficients/exponents
    C_N    = np.maximum(C_N,    1e-12)
    C_D    = np.maximum(C_D,    1e-12)
    alpha  = np.maximum(alpha,  1e-6)
    beta   = np.maximum(beta,   1e-6)
    gamma  = np.maximum(gamma,  1e-6)

    # Compute effective data seen, saturating with repetition:
    # E_data = U * (1 - exp(-gamma * (T/U)))
    # Where T = tokens, U = unique_tokens
    ratio = tokens / np.maximum(unique_tokens, 1e-12)
    E_data = unique_tokens * (1.0 - np.exp(-gamma * ratio))
    E_data = np.maximum(E_data, 1e-8)  # avoid zero

    # Model-size term: N^{-alpha}
    model_term = model_size**(-alpha)

    # Final loss
    loss = a0 + C_N * model_term + C_D * E_data**(-beta)
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fits the scaling law parameters to minimize MSE against observed loss.
    Returns params = [a0, C_N, alpha, C_D, beta, gamma].
    """
    tokens       = np.asarray(tokens,       dtype=float)
    model_size   = np.asarray(model_size,   dtype=float)
    unique_tokens= np.asarray(unique_tokens,dtype=float)
    loss_values  = np.asarray(loss_values,  dtype=float)

    # We optimize in a mixed space: a0 real, others in log-space to enforce positivity
    def unpack(x):
        a0    = x[0]
        C_N   = np.exp(x[1])
        alpha = np.exp(x[2])
        C_D   = np.exp(x[3])
        beta  = np.exp(x[4])
        gamma = np.exp(x[5])
        return np.array([a0, C_N, alpha, C_D, beta, gamma])

    def objective(x):
        params = unpack(x)
        pred   = scaling_law_func(tokens, model_size, unique_tokens, params)
        return np.mean((pred - loss_values)**2)

    # Initial guess: baseline at min observed loss, other logs at moderate values
    a0_init = np.min(loss_values)
    x0 = np.array([
        a0_init,        # a0
        np.log(1.0),    # log C_N
        np.log(0.5),    # log alpha
        np.log(1.0),    # log C_D
        np.log(0.5),    # log beta
        np.log(0.5)     # log gamma
    ])

    # Optimize with L-BFGS-B
    result = minimize(objective, x0, method='L-BFGS-B')
    if result.success:
        return unpack(result.x)
    else:
        # Fallback to initial if optimization fails
        return unpack(x0)

# Number of parameters expected
scaling_law_func.num_params = 6
# EVOLVE-BLOCK-END