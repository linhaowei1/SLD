"""
Domain mixture scaling law discovery for LLM training scenarios
Revised scaling law:
    L_i(r) = c_i + k_i * exp(-t_i * r_i)
Each domain’s loss depends exponentially on its own mixture proportion.
Total parameters: 15 (for each of 5 domains: c_i, k_i, t_i).
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    A revised scaling law function modeling each domain’s loss
    as an exponential decay in its own proportion:
    
        L_i(r) = c_i + k_i * exp(-t_i * r_i)
    
    Params layout (15 total):
      params[0:5]   = c_i  (domain biases, unconstrained)
      params[5:10]  = k_i  (domain scales, unconstrained)
      params[10:15] = t_i  (domain sensitivities, unconstrained)
    
    We square k_i and t_i to enforce positivity:
      k_i_pos = k_i**2, t_i_pos = t_i**2
    
    Args:
        proportions: Array [n_samples, 5], each row sums to 1.0
        params:      Array of length ≥15 (will be padded/truncated to 15)
        
    Returns:
        loss_predictions: Array [n_samples, 5]
    """
    # Ensure 2D array of proportions
    proportions = np.atleast_2d(proportions).astype(float)
    # Prepare parameter vector of length 15
    p = np.asarray(params, dtype=float)
    if p.size < 15:
        p_full = np.ones(15, dtype=float)
        p_full[:p.size] = p
        p = p_full
    else:
        p = p[:15]
    # Unpack raw parameters
    c_raw = p[0:5]    # biases
    k_raw = p[5:10]   # scales (will square)
    t_raw = p[10:15]  # sensitivities (will square)
    # Enforce positivity
    c = c_raw
    k = k_raw**2
    t = t_raw**2
    # Compute domain-wise exponential decay: exp(-t_i * r_i)
    # proportions shape: [n_samples, 5]
    # t shape: [5], broadcasting over columns
    exp_term = np.exp(- proportions * t)
    # Compute losses: c_i + k_i * exp(-t_i * r_i)
    # Broadcasting k over rows
    loss_predictions = c + k * exp_term
    return loss_predictions

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law to domain proportions and loss values
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters: c_i, k_i, t_i)
    """
    # Initialize 15 parameters (5 c's, 5 k's, 5 t's)
    initial_params = np.ones(15)
    # Set c_i to mean loss per domain, k_i=1, t_i=1
    for i in range(5):
        initial_params[i]       = np.mean(loss_values[:, i])  # c_i
        initial_params[i + 5]   = 1.0                         # k_i
        initial_params[i + 10]  = 1.0                         # t_i
    
    def objective(params):
        try:
            pred = scaling_law_func(proportions, params)
            return np.mean((pred - loss_values) ** 2)
        except:
            return 1e6
    
    best_params = initial_params.copy()
    best_loss = float('inf')
    
    # Three BFGS attempts
    for attempt in range(3):
        if attempt == 0:
            start = initial_params
        else:
            start = initial_params + np.random.randn(15) * 0.1
        try:
            res = minimize(objective, start, method='BFGS')
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            pass
    
    return best_params

# Number of parameters this scaling law expects
scaling_law_func.num_params = 15