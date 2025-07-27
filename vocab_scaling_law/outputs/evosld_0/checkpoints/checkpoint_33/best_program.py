import numpy as np

# EVOLVE-BLOCK-START
def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Predict Lossu via a 7-parameter quadratic model in log-space:

    Let:
        P = Non_vocab_parameters
        V = vocab_size
        C = num_characters

    Define:
        lP = log(P_clamped)
        lV = log(V_clamped)
        lC = log(C_clamped)

    Then:
        Lossu ≈ θ0
               + θ1 * lP
               + θ2 * lC
               + θ3 * lV
               + θ4 * (lP)^2
               + θ5 * (lC)^2
               + θ6 * (lV)^2

    This captures individual curvature in each axis while keeping parameter count = 7.
    """
    # clamp to avoid log(0) and extremely small values
    P = np.maximum(Non_vocab_parameters, 1.0)
    V = np.maximum(vocab_size, 1.0)
    C = np.maximum(num_characters, 1.0)

    lP = np.log(P)
    lC = np.log(C)
    lV = np.log(V)

    # build design matrix (n_samples x 7)
    X = np.stack([
        np.ones_like(lP),
        lP,
        lC,
        lV,
        lP * lP,
        lC * lC,
        lV * lV
    ], axis=-1)

    # linear model
    return X.dot(params)


def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter quadratic log-space scaling law via ridge‐regularized least squares.
    Returns optimized params [θ0 … θ6].
    """
    # clamp inputs
    P = np.maximum(Non_vocab_parameters, 1.0)
    V = np.maximum(vocab_size, 1.0)
    C = np.maximum(num_characters, 1.0)

    lP = np.log(P)
    lC = np.log(C)
    lV = np.log(V)

    # assemble design matrix X (n_samples x 7)
    X = np.vstack([
        np.ones_like(lP),
        lP,
        lC,
        lV,
        lP * lP,
        lC * lC,
        lV * lV
    ]).T

    y = lossu_values

    # small ridge regularization for stability
    ridge_strength = 1e-5
    G = X.T @ X
    # add ridge on diagonals
    diag_idx = np.diag_indices_from(G)
    G[diag_idx] += ridge_strength
    rhs = X.T @ y

    # solve normal equations
    params = np.linalg.solve(G, rhs)
    return params

# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END