# EVOLVE-BLOCK-START
"""
Refined data‐constrained scaling law for LLM training loss.
Uses 7 parameters:
  a     : asymptotic minimum loss
  b     : coefficient of model‐size term
  c     : coefficient of data‐term
  alpha : model‐size exponent (>0)
  beta  : data‐term exponent (>0)
  k     : relative saturation scale (>0)
  s     : shape exponent for data saturation (>0)
Model:
  T = tokens/1e9, N = model_size/1e9, U = unique_tokens/1e9
  E = U * (1 - exp(-(T/(k*U))**s))
  L = a + b * N**(-alpha) + c * E**(-beta)
Fitted via multi‐start bounded L-BFGS minimizing MSE + 0.1·log‐MSE.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    a, b, c, alpha, beta, k, s = params
    eps = 1e-12
    # normalize to billions for stability
    T = np.maximum(tokens, 0.0) / 1e9
    N = np.maximum(model_size, eps) / 1e9
    U = np.maximum(unique_tokens, eps) / 1e9
    # effective data with shape exponent s
    ratio = T / (k * U + eps)
    E = U * (1.0 - np.exp(-np.power(ratio, s)))
    # model and data power-law terms
    N_term = np.power(N + eps, -alpha)
    E_term = np.power(E + eps, -beta)
    # predict loss
    return a + b * N_term + c * E_term

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    tokens = np.asarray(tokens, dtype=float)
    model_size = np.asarray(model_size, dtype=float)
    unique_tokens = np.asarray(unique_tokens, dtype=float)
    loss_values = np.asarray(loss_values, dtype=float)

    Lmin, Lmax = loss_values.min(), loss_values.max()
    span = max(Lmax - Lmin, 1e-3)

    # parameter bounds
    bounds = [
        (0.0,        Lmax),      # a
        (1e-6,      100*span),   # b
        (1e-6,      100*span),   # c
        (1e-2,        3.0),      # alpha
        (1e-2,        3.0),      # beta
        (1e-2,       20.0),      # k
        (1e-2,        5.0)       # s
    ]

    # default initial guess
    init = np.array([
        Lmin,            # a
        0.5*span,        # b
        0.5*span,        # c
        0.7,             # alpha
        0.7,             # beta
        1.0,             # k
        1.0              # s
    ], dtype=float)

    # objective combining MSE and log‐MSE
    def objective(p):
        pred = scaling_law_func(tokens, model_size, unique_tokens, p)
        mse = np.mean((pred - loss_values)**2)
        lp = np.log(np.maximum(pred, 1e-8))
        lt = np.log(np.maximum(loss_values, 1e-8))
        lmse = np.mean((lp - lt)**2)
        return mse + 0.1 * lmse

    # multi‐start optimization
    best_obj = np.inf
    best_params = init.copy()
    rng = np.random.RandomState(12345)

    # generate 6 starting points (1 default + 5 random)
    inits = [init]
    for _ in range(5):
        rand_p = []
        for (low, high) in bounds:
            rand_p.append(rng.uniform(low, high))
        inits.append(np.array(rand_p, dtype=float))

    for x0 in inits:
        res = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x

    return best_params

# record number of parameters used
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END