# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def scaling_law_func(proportions, params):
    """
    Predict per-domain losses given mixture proportions and parameters.
    Model for each domain i:
       L_i = α_i
            + sum_{j=1..5} β_{i,j} * [ -log(p_j + ε) ]^{γ_i}
    Total parameters: 7 per domain (α_i, γ_i, β_{i,1..5}) => 35.
    Args:
        proportions: array-like, shape (n_samples, 5), rows sum to 1
        params:       array-like, length 35, flattened domain blocks
    Returns:
        losses_pred:  ndarray, shape (n_samples, 5)
    """
    p = np.atleast_2d(proportions).astype(float)
    n_samples, n_dom = p.shape
    assert n_dom == 5, "Expected 5 domain proportions"
    eps = 1e-8
    # compute x = -log(p + eps)
    x = -np.log(p + eps)                       # (n_samples, 5)

    P = np.ravel(params)
    assert P.size == 35, f"Expected 35 params, got {P.size}"
    P = P.reshape(5, 7)                        # one row per domain
    alpha = P[:, 0]                            # (5,)
    gamma = P[:, 1]                            # (5,)
    beta  = P[:, 2:]                           # (5,5)

    # raise x to each domain's gamma: shape -> (n_samples, 5_targets, 5_inputs)
    # x_expanded: (n_samples, 1, 5), gamma_exp: (1,5,1)
    x_exp = x[:, None, :]                      # (n_samples,1,5)
    g_exp = gamma[None, :, None]               # (1,5,1)
    x_pow = x_exp ** g_exp                     # (n_samples,5,5)

    # weight by beta: beta shape (5_targets,5_inputs) -> (1,5,5)
    b_exp = beta[None, :, :]                   # (1,5,5)
    weighted = x_pow * b_exp                   # (n_samples,5,5)

    # sum over input domains j
    sum_term = weighted.sum(axis=2)            # (n_samples,5)

    # add intercept alpha
    losses = sum_term + alpha[None, :]         # (n_samples,5)
    return losses

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law parameters domain-wise via nonlinear least squares.
    Returns flattened params length 35.
    """
    p = np.atleast_2d(proportions).astype(float)
    y = np.atleast_2d(loss_values).astype(float)
    n_samples, n_dom = p.shape
    assert n_dom == 5 and y.shape == (n_samples, 5)

    eps = 1e-8
    x = -np.log(p + eps)   # (n_samples,5)

    all_params = []

    for i in range(5):
        yi = y[:, i]       # (n_samples,)
        xi = x             # reuse same x for all domains

        # initial linear fit for β_{i,*} and α_i
        #   solve [xi | 1] @ [β; α] ≈ yi
        A = np.concatenate([xi, np.ones((n_samples, 1))], axis=1)  # (n_samples,6)
        try:
            sol_lin, *_ = np.linalg.lstsq(A, yi, rcond=None)
            beta0 = sol_lin[:5]
            alpha0 = sol_lin[5]
        except Exception:
            alpha0 = np.median(yi)
            beta0  = np.zeros(5)

        gamma0 = 1.0
        p0 = np.concatenate([[alpha0, gamma0], beta0])  # length 7

        # bounds: α free, γ in [0.01,5], β free
        lower = np.array([-np.inf, 0.01] + [-np.inf]*5)
        upper = np.array([ np.inf, 5.00 ] + [ np.inf]*5)

        def resid(params_i):
            ai = params_i[0]
            gi = params_i[1]
            bi = params_i[2:]
            # compute sum_j β_{ij} * x_j^{γ_i}
            xp = xi ** gi                           # (n_samples,5)
            pred = ai + xp.dot(bi)                  # (n_samples,)
            return pred - yi

        sol = least_squares(resid, p0, bounds=(lower, upper),
                            xtol=1e-8, ftol=1e-8, verbose=0)
        all_params.append(sol.x)

    return np.concatenate(all_params)

# expose parameter count
scaling_law_func.num_params = 35
# EVOLVE-BLOCK-END