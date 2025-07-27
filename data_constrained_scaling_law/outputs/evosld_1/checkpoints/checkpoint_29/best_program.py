# EVOLVE-BLOCK-START
"""
Evolved data‐constrained scaling law for LLM training scenarios with
saturating effective‐data term and linear coefficient inner‐solve.

We model:
  T_eff = unique_tokens * (1 - exp(- tokens / (unique_tokens * θ)))
  M_norm = model_size / 1e9
  E_norm = T_eff / 1e9

Scaling law (6 params):
   loss_pred = A
             + B * M_norm^(−α)
             + C * E_norm^(−β)

where params = [A, B, α, C, β, θ].

We fit {α, β, θ} by minimizing a combined MSE + 0.05·log‐MSE,
and for each trial compute optimal (A, B, C) via linear least‐squares.
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(tokens, model_size, unique_tokens, params):
    """
    tokens:       (N,) array of training tokens used
    model_size:   (N,) array of model parameter counts
    unique_tokens:(N,) array of unique tokens available
    params:       [A, B, α, C, β, θ]
    returns:      (N,) predicted loss
    """
    A, B, alpha, C, beta, theta = params
    # normalize model size
    M_norm = np.maximum(model_size / 1e9, 1e-6)
    # effective data saturation
    # clamp unique_tokens to avoid zero
    U = np.maximum(unique_tokens, 1.0)
    ratio = tokens / (U * theta + 0.0)
    # ensure numeric stability
    ratio = np.minimum(np.maximum(ratio, 0.0), 1e3)
    T_eff = U * (1.0 - np.exp(-ratio))
    E_norm = np.maximum(T_eff / 1e9, 1e-6)

    return A + B * (M_norm ** (-alpha)) + C * (E_norm ** (-beta))

# annotate number of parameters
scaling_law_func.num_params = 6

def fit_scaling_law(tokens, model_size, unique_tokens, loss_values):
    """
    Fit the scaling law with inner linear solve for [A,B,C] and
    outer optimization over [α, β, θ]:
      - minimize MSE + 0.05*log‐MSE
      - solve for A,B,C in closed form each step
    Returns: params = [A, B, α, C, β, θ]
    """
    # cast to arrays
    T = np.asarray(tokens, dtype=float)
    M = np.asarray(model_size, dtype=float)
    U = np.asarray(unique_tokens, dtype=float)
    Y = np.asarray(loss_values, dtype=float)
    # pre‐normalize M_norm
    M_norm = np.maximum(M / 1e9, 1e-6)
    # stability epsilon
    eps = 1e-12

    # objective for outer params x = [u, v, w] where
    #   α = exp(u), β = exp(v), θ = exp(w)
    def outer_obj(x):
        u, v, w = x
        alpha = np.exp(u)
        beta = np.exp(v)
        theta = np.exp(w)

        # compute effective tokens
        Uc = np.maximum(U, 1.0)
        ratio = T / (Uc * theta)
        # clamp ratio
        ratio = np.minimum(np.maximum(ratio, 0.0), 1e3)
        T_eff = Uc * (1.0 - np.exp(-ratio))
        E_norm = np.maximum(T_eff / 1e9, 1e-6)

        # design matrix: columns [1, M_norm^-α, E_norm^-β]
        X = np.vstack([
            np.ones_like(Y),
            M_norm ** (-alpha),
            E_norm ** (-beta),
        ]).T  # shape (N,3)

        # solve linear least squares for coeffs [A, B, C]
        # we do unconstrained; will clamp negatives
        coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
        A, B, C = coeffs
        # clamp to non‐negative minima
        A = max(A, eps)
        B = max(B, eps)
        C = max(C, eps)

        # predictions
        Ypred = A + B * (M_norm ** (-alpha)) + C * (E_norm ** (-beta))
        # compute MSE
        err = Ypred - Y
        mse = np.mean(err * err)
        # log‐mse
        lp = np.log(np.maximum(Ypred, eps))
        lt = np.log(np.maximum(Y, eps))
        logmse = np.mean((lp - lt) ** 2)

        return mse + 0.05 * logmse

    # initial logs: α,β ~0.5, θ ~1.0
    init = np.log([0.5, 0.5, 1.0])
    # bounds on u,v,w
    bnds = [
        (np.log(1e-2), np.log(5.0)),   # u = log(α)
        (np.log(1e-2), np.log(5.0)),   # v = log(β)
        (np.log(1e-2), np.log(1e1)),   # w = log(θ)
    ]

    best = None
    best_val = np.inf
    # multi‐start
    for scale in [0.8, 1.0, 1.2]:
        x0 = init * scale
        res = minimize(
            outer_obj,
            x0,
            method="L-BFGS-B",
            bounds=bnds,
            options={"ftol":1e-8, "maxiter":1000}
        )
        if res.success and res.fun < best_val:
            best = res
            best_val = res.fun

    # fallback to init if needed
    if best is None:
        best = res

    # recover best outer params
    u_opt, v_opt, w_opt = best.x
    α_opt = np.exp(u_opt)
    β_opt = np.exp(v_opt)
    θ_opt = np.exp(w_opt)

    # compute final A,B,C
    Uc = np.maximum(U, 1.0)
    ratio = T / (Uc * θ_opt)
    ratio = np.minimum(np.maximum(ratio, 0.0), 1e3)
    T_eff = Uc * (1.0 - np.exp(-ratio))
    E_norm = np.maximum(T_eff / 1e9, 1e-6)
    X_final = np.vstack([
        np.ones_like(Y),
        M_norm ** (-α_opt),
        E_norm ** (-β_opt),
    ]).T
    coeffs, *_ = np.linalg.lstsq(X_final, Y, rcond=None)
    A_opt, B_opt, C_opt = coeffs
    A_opt = max(A_opt, eps)
    B_opt = max(B_opt, eps)
    C_opt = max(C_opt, eps)

    return np.array([A_opt, B_opt, α_opt, C_opt, β_opt, θ_opt])

# EVOLVE-BLOCK-END