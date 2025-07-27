# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Quadratic-in-log scaling law (7 parameters):

    Let
        P = log(Non_vocab_parameters)
        V = log(vocab_size)
        N = log(num_characters)

    Then
        Lossu ≈ θ0
               + θ1 * P
               + θ2 * V
               + θ3 * N
               + θ4 * P**2
               + θ5 * V**2
               + θ6 * N**2

    This captures nonlinearity in each axis while remaining
    linear in parameters for closed-form fitting.
    """
    θ0, θ1, θ2, θ3, θ4, θ5, θ6 = params

    # ensure positivity inside log
    P = np.log(np.maximum(Non_vocab_parameters, 1.0))
    V = np.log(np.maximum(vocab_size,          1.0))
    N = np.log(np.maximum(num_characters,      1.0))

    return (θ0
            + θ1 * P
            + θ2 * V
            + θ3 * N
            + θ4 * P * P
            + θ5 * V * V
            + θ6 * N * N)

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter quadratic-in-log scaling law via least squares:

        params = argmin ||X·params - y||^2

    where X columns are [1, P, V, N, P^2, V^2, N^2].
    """
    # flatten and compute logs
    P = np.log(np.maximum(Non_vocab_parameters.ravel(), 1.0))
    V = np.log(np.maximum(vocab_size.ravel(),          1.0))
    N = np.log(np.maximum(num_characters.ravel(),      1.0))
    y = lossu_values.ravel()

    # build design matrix
    X = np.vstack([
        np.ones_like(P),
        P,
        V,
        N,
        P * P,
        V * V,
        N * N
    ]).T  # shape (n_samples, 7)

    # closed-form least squares solution
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    return params

# annotate the number of parameters
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END