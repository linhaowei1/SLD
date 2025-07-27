# EVOLVE-BLOCK-START
import numpy as np

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Log-linear scaling law with pairwise interactions (7 parameters total):
    
    Lossu ≈ θ0 
           + θ1 * log(Non_vocab_parameters)
           + θ2 * log(vocab_size)
           + θ3 * log(num_characters)
           + θ4 * log(Non_vocab_parameters) * log(vocab_size)
           + θ5 * log(Non_vocab_parameters) * log(num_characters)
           + θ6 * log(vocab_size) * log(num_characters)
    """
    # Unpack parameters
    θ0, θ1, θ2, θ3, θ4, θ5, θ6 = params

    # Safely compute logs
    P = np.log(np.maximum(Non_vocab_parameters, 1.0))
    V = np.log(np.maximum(vocab_size, 1.0))
    N = np.log(np.maximum(num_characters, 1.0))

    return (θ0
            + θ1 * P
            + θ2 * V
            + θ3 * N
            + θ4 * P * V
            + θ5 * P * N
            + θ6 * V * N)

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the 7-parameter log-linear scaling law via closed-form least squares.

    Returns:
        params: array of shape (7,) corresponding to
                [θ0, θ1, θ2, θ3, θ4, θ5, θ6]
    """
    # Flatten inputs
    P = np.log(np.maximum(Non_vocab_parameters.ravel(), 1.0))
    V = np.log(np.maximum(vocab_size.ravel(), 1.0))
    N = np.log(np.maximum(num_characters.ravel(), 1.0))
    y = lossu_values.ravel()

    # Construct design matrix X: [1, P, V, N, P*V, P*N, V*N]
    X = np.vstack([
        np.ones_like(P),
        P,
        V,
        N,
        P * V,
        P * N,
        V * N
    ]).T  # shape (n_samples, 7)

    # Solve normal equations via least squares
    # params = argmin ||X·params - y||^2
    params, *_ = np.linalg.lstsq(X, y, rcond=None)  # returns (7,), residuals, rank, s

    return params

# annotate expected parameter count
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END