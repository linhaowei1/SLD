import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def scaling_law_func(Non_vocab_parameters, vocab_size, num_characters, params):
    """
    Quadratic interaction model in normalized log-space (7 parameters):

      Let
        x = log10(Non_vocab_parameters)
        y = log10(vocab_size)
        z = log10(num_characters)

      Normalize each to zero mean and unit range:
        x' = (x - mean(x)) / (max(x)-min(x))
        y' = (y - mean(y)) / (max(y)-min(y))
        z' = (z - mean(z)) / (max(z)-min(z))

      Then model Lossu ≈ p0
                         + p1·x'
                         + p2·y'
                         + p3·z'
                         + p4·(x'·y')
                         + p5·(x'·z')
                         + p6·(y'·z')

    params layout:
      params[0] = p0 (intercept)
      params[1] = p1 (coef for x')
      params[2] = p2 (coef for y')
      params[3] = p3 (coef for z')
      params[4] = p4 (coef for x'·y')
      params[5] = p5 (coef for x'·z')
      params[6] = p6 (coef for y'·z')
    """
    p0, p1, p2, p3, p4, p5, p6 = params

    # take base-10 logs
    x = np.log10(Non_vocab_parameters)
    y = np.log10(vocab_size)
    z = np.log10(num_characters)

    # normalize each feature to zero mean and unit range
    # (range = max - min)
    xr = (x - np.mean(x)) / (np.max(x) - np.min(x) + 1e-12)
    yr = (y - np.mean(y)) / (np.max(y) - np.min(y) + 1e-12)
    zr = (z - np.mean(z)) / (np.max(z) - np.min(z) + 1e-12)

    # build quadratic interaction terms
    xy = xr * yr
    xz = xr * zr
    yz = yr * zr

    # linear + interaction terms
    return (
        p0
        + p1 * xr
        + p2 * yr
        + p3 * zr
        + p4 * xy
        + p5 * xz
        + p6 * yz
    )
# EVOLVE-BLOCK-END

def fit_scaling_law(Non_vocab_parameters, vocab_size, num_characters, lossu_values):
    """
    Fit the scaling law to vocabulary data and Lossu values
    
    Args:
        Non_vocab_parameters: Array of non-vocabulary parameter counts
        vocab_size: Array of vocabulary sizes
        num_characters: Array of number of characters processed
        lossu_values: Array of corresponding Lossu values
        
    Returns:
        Optimized parameters (7 parameters)
    """
    # Initialize parameters with ones
    initial_params = np.ones(7)
    
    def objective(params):
        try:
            predicted = scaling_law_func(
                Non_vocab_parameters, vocab_size, num_characters, params
            )
            mse = np.mean((predicted - lossu_values) ** 2)
            return mse
        except:
            return 1e6  # Large penalty on failure
    
    result = minimize(objective, initial_params, method='BFGS')
    final_params = result.x if result.success else initial_params
    return final_params

# Set the number of parameters this function expects
scaling_law_func.num_params = 7