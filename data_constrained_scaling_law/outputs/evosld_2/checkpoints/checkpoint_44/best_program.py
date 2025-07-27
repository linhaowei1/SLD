# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    7-parameter data-constrained scaling law:
      L_pred = L_inf
             + A * M_norm^(-alpha)
             + B * D_norm^(-beta)

    where
      M_norm = model_size / 1e9
      U_norm = unique_tokens / 1e9
      ratio  = tokens / (c * unique_tokens + eps)
      D_norm = U_norm * (1 - exp(- ratio^gamma))

    params = [L_inf, A, alpha, B, beta, c, gamma]
    """
    # unpack parameters
    L_inf, A, alpha, B, beta, c, gamma = params

    # to float arrays
    T = np.asarray(tokens, dtype=np.float64)
    M = np.asarray(model_size, dtype=np.float64)
    U = np.asarray(unique_tokens, dtype=np.float64)
    eps = 1e-12

    # normalized model size and unique tokens
    M_norm = M / 1e9 + eps
    U_norm = U / 1e9 + eps

    # effective data coverage with learnable saturation c and shape gamma
    ratio = T / (c * U + eps)
    D_norm = U_norm * (1 - np.exp(-np.power(ratio, gamma)))

    # final predicted loss
    return L_inf \
           + A * np.power(M_norm, -alpha) \
           + B * np.power(D_norm + eps, -beta)


def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the 7-parameter scaling law by minimizing mean squared error.
    Returns optimized params = [L_inf, A, alpha, B, beta, c, gamma].
    """
    # prepare data arrays
    T = np.asarray(tokens, dtype=np.float64)
    M = np.asarray(model_size, dtype=np.float64)
    U = np.asarray(unique_tokens, dtype=np.float64)
    L = np.asarray(loss_values, dtype=np.float64)

    # initial guess based on observed loss range
    L0      = max(1e-6, np.min(L) * 0.9)
    range_L = max(1e-6, np.max(L) - np.min(L))
    x0 = np.array([
        L0,            # L_inf
        range_L,       # A
        0.5,           # alpha
        range_L * 0.5, # B
        0.5,           # beta
        1.0,           # c
        0.5            # gamma
    ], dtype=np.float64)

    # bounds for stability
    bnds = [
        (0.0,         np.min(L)),  # L_inf <= min observed loss
        (1e-8,        None),       # A > 0
        (1e-6,        5.0),        # alpha in [1e-6,5]
        (1e-8,        None),       # B > 0
        (1e-6,        5.0),        # beta in [1e-6,5]
        (1e-6,        1e2),        # c in [1e-6,1e2]
        (1e-6,        10.0)        # gamma in [1e-6,10]
    ]

    # objective: mean squared error
    def objective(p):
        pred = scaling_law_func(T, M, U, p)
        return np.mean((pred - L)**2)

    # optimize
    res = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    return res.x if res.success else x0

# annotate number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END