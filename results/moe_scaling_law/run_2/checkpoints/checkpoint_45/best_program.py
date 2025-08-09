import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(data_points, params):
    """
    Predict validation loss L given:
      data_points: array-like of shape (N,2) [num_experts E, dense_param_count P]
      params:      array-like of length 6 [θ0, θ1, θ2, θ3, θ4, θ5]

    Model in log‐space:
      ln L = θ0
           + θ1·ln(P)
           + θ2·ln(E)
           + θ3·[ln(P) * ln(E)]
           + θ4·(1/P)
           + θ5·[1/(P·E)]

    Returns:
      preds: ndarray of shape (N,) with predicted losses.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 2:
        raise ValueError(f"data_points must have shape (N,2), got {X.shape}")
    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)

    lnE = np.log(E)
    lnP = np.log(P)
    invP = 1.0 / P
    invPE = invP / E  # 1/(P·E)

    θ = np.asarray(params, dtype=float).ravel()
    if θ.size != 6:
        raise ValueError(f"Expected 6 parameters, got {θ.size}")
    θ0, θ1, θ2, θ3, θ4, θ5 = θ

    lnL = (
        θ0
        + θ1 * lnP
        + θ2 * lnE
        + θ3 * (lnP * lnE)
        + θ4 * invP
        + θ5 * invPE
    )
    return np.exp(lnL)


def fit_scaling_law(data_points, loss_values):
    """
    Fit the 6-parameter model by ordinary least squares on ln(loss):
      ln(y) ≈ θ0·1 + θ1·ln(P) + θ2·ln(E) + θ3·[ln(P)·ln(E)]
                       + θ4·(1/P) + θ5·[1/(P·E)]

    Returns:
      θ_opt: ndarray of shape (6,) with the fitted parameters.
    """
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    if X.shape[1] != 2:
        raise ValueError(f"data_points must have shape (N,2), got {X.shape}")
    y = np.asarray(loss_values, dtype=float).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError("loss_values must have the same number of rows as data_points")

    # Avoid log(0) or negative
    y_safe = np.clip(y, 1e-8, None)
    lnY = np.log(y_safe)

    E = np.clip(X[:, 0], 1e-8, None)
    P = np.clip(X[:, 1], 1e-8, None)
    lnE = np.log(E)
    lnP = np.log(P)
    invP = 1.0 / P
    invPE = invP / E

    # Design matrix: [1, lnP, lnE, lnP·lnE, 1/P, 1/(P·E)]
    M = np.column_stack([
        np.ones_like(lnP),
        lnP,
        lnE,
        lnP * lnE,
        invP,
        invPE
    ])

    θ_opt, *_ = np.linalg.lstsq(M, lnY, rcond=None)
    return θ_opt
# EVOLVE-BLOCK-END