# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    A normalized log‐interaction model with 7 parameters:
      Lossu ≈ p0
            + p1·xV
            + p2·xN
            + p3·xC
            + p4·xV·xN
            + p5·xV·xC
            + p6·xN·xC

    where xV, xN, xC are the standardized logs of
    vocab_size, Non_vocab_parameters, and num_characters.

    Args:
        Non_vocab_parameters: Array of non‐vocabulary parameter counts
        vocab_size:           Array of vocabulary sizes  
        num_characters:       Array of number of characters processed
        params:               Array of 7 parameters [p0 ... p6]

    Returns:
        Predicted Lossu values as a NumPy array
    """
    # tiny epsilon to avoid log(0)
    eps = 1e-8

    # raw logs
    logV = np.log(vocab_size + eps)
    logN = np.log(Non_vocab_parameters + eps)
    logC = np.log(num_characters + eps)

    # standardize each log‐feature (zero mean, unit variance)
    meanV, stdV = logV.mean(), logV.std() + eps
    meanN, stdN = logN.mean(), logN.std() + eps
    meanC, stdC = logC.mean(), logC.std() + eps

    xV = (logV - meanV) / stdV
    xN = (logN - meanN) / stdN
    xC = (logC - meanC) / stdC

    # pairwise interactions in normalized space
    xVxN = xV * xN
    xVxC = xV * xC
    xNxC = xN * xC

    # linear combination of the seven features
    # params = [p0, p1, p2, p3, p4, p5, p6]
    lossu_pred = (
        params[0]
        + params[1] * xV
        + params[2] * xN
        + params[3] * xC
        + params[4] * xVxN
        + params[5] * xVxC
        + params[6] * xNxC
    )
    return lossu_pred

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to vocabulary data and Lossu values using BFGS.
    """
    # Initialize parameters (7 parameters for the normalized interaction model)
    initial_params = np.ones(7)

    def objective(params):
        try:
            pred = scaling_law_func(
                Non_vocab_parameters, vocab_size, num_characters, params
            )
            return np.mean((pred - lossu_values) ** 2)
        except:
            return 1e6

    result = minimize(objective, initial_params, method='BFGS')
    return result.x if result.success else initial_params

# metadata for parameter counting
scaling_law_func.num_params = 7
# EVOLVE-BLOCK-END