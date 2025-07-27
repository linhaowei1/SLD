import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def scaling_law_func(proportions, params):
    """
    New form (7 params per domain):
      α_i      : bias
      γ_i      : exponent on log‐feature
      β_i1..β_i5 : weights on x_j = -log(p_j+ε)

    Li = α_i + sum_{j=1..5} β_ij * ( x_j )^(γ_i)

    params is flat length 35, reshaped to (5,7):
      [α, γ, β1..β5] for each domain
    """
    P = np.atleast_2d(proportions).astype(float)
    n, d = P.shape
    assert d == 5, "need 5 domains"
    mat = params.reshape(5, 7)
    alphas = mat[:, 0]        # (5,)
    gammas = mat[:, 1]        # (5,)
    betas  = mat[:, 2:]       # (5,5)

    eps = 1e-10
    X = -np.log(P + eps)      # shape (n,5)
    L = np.zeros((n, 5), dtype=float)

    # for each domain i, raise all x_j to γ_i, then dot with β_i
    for i in range(5):
        Xi = X ** gammas[i]      # (n,5)
        L[:, i] = alphas[i] + Xi.dot(betas[i])
    return L

def fit_scaling_law(proportions, loss_values):
    """
    1) Quick linear least‐squares to get α_i, β_ij for γ_i=1
    2) Joint nonlinear refinement of all 35 params
    """
    P = np.atleast_2d(proportions).astype(float)
    Y = np.atleast_2d(loss_values).astype(float)
    n, d = P.shape
    assert d == 5 and Y.shape == (n, 5)

    # Precompute log‐features
    eps = 1e-10
    X = -np.log(P + eps)      # (n,5)
    A = np.hstack([np.ones((n, 1)), X])  # for linear fit: [1, x1..x5]

    # Build initial parameter vector
    init = np.zeros(5 * 7, dtype=float)
    for i in range(5):
        # solve Li ≈ α + ∑ β_j x_j  via least‐squares
        sol, *_ = np.linalg.lstsq(A, Y[:, i], rcond=None)
        α_i   = max(sol[0], 0.0)
        β_i   = np.clip(sol[1:], 1e-6, None)
        # pack: [α, γ, β1..β5]
        init[7 * i + 0] = α_i
        init[7 * i + 1] = 1.0       # start γ_i at 1
        init[7 * i + 2:7 * i + 7] = β_i

    # bounds: α>=0, γ in [0.1,10], β>=0
    lower = np.zeros_like(init)
    upper = np.full_like(init, np.inf)
    for i in range(5):
        lower[7*i+0] = 0.0
        lower[7*i+1] = 0.1
        upper[7*i+1] = 10.0
        lower[7*i+2:7*i+7] = 0.0

    def residuals(p):
        return (scaling_law_func(P, p) - Y).ravel()

    res = least_squares(
        residuals,
        init,
        bounds=(lower, upper),
        method='trf',
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=3000
    )
    return res.x
# EVOLVE-BLOCK-END

# metadata
scaling_law_func.num_params = 35