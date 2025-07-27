# EVOLVE-BLOCK-START
"""
Enhanced data‐constrained scaling law for LLM training loss,
introducing a flexible saturation exponent and a mixed‐objective fit
for better absolute and relative error trade‐offs.
Uses at most 7 parameters.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    Predict training loss given:
      tokens       : total training tokens used
      model_size   : number of model parameters
      unique_tokens: size of unique data vocabulary
      params       : [a, b, c, alpha, beta, k, gamma]
        a     : asymptotic minimum loss
        b     : model-size coefficient
        c     : data-size coefficient
        alpha : model-size exponent (>0)
        beta  : data-size exponent (>0)
        k     : data-saturation scale (>0)
        gamma : saturation shape exponent (>0)
    """
    a, b, c, alpha, beta, k, gamma = params

    # clip and rescale for numerical stability
    T = np.maximum(tokens,       0.0) / 1e7   # scaled total tokens
    N = np.maximum(model_size,   1.0) / 1e7   # scaled model size
    U = np.maximum(unique_tokens,1.0) / 1e7   # scaled unique vocab

    # flexible data saturation:
    # ratio = T / (k * U)  --> may be >0
    ratio = T / (k * U + 1e-12)
    E = U * (1.0 - np.exp(-np.power(ratio, gamma)))
    E = np.maximum(E, 1e-8)

    # final loss: asymptote + model term + data term
    loss = a + b * np.power(N, -alpha) + c * np.power(E, -beta)
    return loss

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7 parameters of the scaling law by minimizing a 
    combined absolute & relative error objective.
    Returns:
      params : array of 7 fitted parameters [a,b,c,alpha,beta,k,gamma]
    """
    # basic stats
    Lmin, Lmax = float(np.min(loss_values)), float(np.max(loss_values))

    # initial guesses
    initial = np.array([
        Lmin,                   # a
        (Lmax - Lmin) * 0.6,    # b
        (Lmax - Lmin) * 0.4,    # c
        0.5,                    # alpha
        0.5,                    # beta
        1.0,                    # k
        1.0                     # gamma
    ], dtype=float)

    # parameter bounds for stability
    bounds = [
        (0.0,         Lmax),                 # a
        (1e-8,       10*(Lmax-Lmin)+1e-6),   # b
        (1e-8,       10*(Lmax-Lmin)+1e-6),   # c
        (1e-3,        3.0),                  # alpha
        (1e-3,        3.0),                  # beta
        (1e-3,       10.0),                  # k
        (1e-3,        5.0)                   # gamma
    ]

    # combined objective: absolute MSE + λ*relative (log) MSE
    def objective(p):
        pred = scaling_law_func(tokens, model_size, unique_tokens, p)
        # absolute MSE
        mse = np.mean((pred - loss_values)**2)
        # log‐space (relative) MSE
        lp = np.log(np.maximum(pred,       1e-8))
        lt = np.log(np.maximum(loss_values,1e-8))
        lmse = np.mean((lp - lt)**2)
        # balance factors (tune λ to prioritize absolute vs. relative)
        return mse + 0.2 * lmse

    # run bounded L-BFGS optimization
    result = minimize(
        objective,
        x0=initial,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-9}
    )

    if result.success:
        return result.x
    else:
        # fallback to initial guess if optimization fails
        return initial

# record number of parameters used
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END