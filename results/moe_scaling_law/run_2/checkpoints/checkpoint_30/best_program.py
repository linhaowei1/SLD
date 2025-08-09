import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Parametric scaling law:
      ln L = θ0 
           + θ1 * ln(P) 
           + θ2 * ln(E) 
           + θ3 * (1 / P) 
           + θ4 * (1 / E) 
           + θ5 * (1 / (P * E))
    => L = exp( ln L ).

    Inputs:
      data_points : (N,2) array of [E, P]
      params       : array-like of length 6 [θ0…θ5]
    Returns:
      preds        : (N,) array of predicted losses.
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    E = X[:, 0]
    P = X[:, 1]

    # numerical safeguards
    E = np.clip(E, 1e-8, None)
    P = np.clip(P, 1e-8, None)

    lnE = np.log(E)
    lnP = np.log(P)
    invE = 1.0 / E
    invP = 1.0 / P
    invPE = invP * invE

    θ = np.asarray(params, dtype=float).ravel()
    if θ.size != 6:
        raise ValueError(f"Expected 6 parameters, got {θ.size}")
    θ0, θ1, θ2, θ3, θ4, θ5 = θ

    u = (θ0
         + θ1 * lnP
         + θ2 * lnE
         + θ3 * invP
         + θ4 * invE
         + θ5 * invPE)
    return np.exp(u)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter model by linear least squares on ln(loss):
      ln y ≈ θ0·1 + θ1·lnP + θ2·lnE + θ3·(1/P) + θ4·(1/E) + θ5·(1/(P·E))

    Returns the optimal θ of shape (6,).
    """
    X = np.asarray(data_points, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y = np.asarray(loss_values, dtype=float).ravel()

    # to keep ln well‐defined
    y = np.clip(y, 1e-8, None)
    lnY = np.log(y)

    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    invE = 1.0 / E
    invP = 1.0 / P
    invPE = invP * invE

    # build design matrix [1, lnP, lnE, 1/P, 1/E, 1/(P*E)]
    M = np.vstack([
        np.ones_like(lnP),
        lnP,
        lnE,
        invP,
        invE,
        invPE
    ]).T

    θ_opt, *_ = np.linalg.lstsq(M, lnY, rcond=None)
    return θ_opt
# EVOLVE-BLOCK-END