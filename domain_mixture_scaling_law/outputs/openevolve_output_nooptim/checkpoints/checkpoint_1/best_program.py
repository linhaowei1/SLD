# EVOLVE-BLOCK-START
"""
Domain mixture scaling law discovery for LLM training scenarios
Improved mixture-exponential form:
    L_i(r) = c_i + k_i * exp(sum_j t_j * r_j)
with 5 domain-specific biases c_i, 5 domain-specific scales k_i,
and 5 global mixture sensitivities t_j.
Total parameters: 15 (5 + 5 + 5).
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(proportions, params):
    """
    A scaling law function to model the relationship between domain proportions
    and loss for each domain using an exponential mixture:
    
    L_i(r) = c_i + k_i * exp(sum_j t_j * r_j)
    
    Params layout (15 total):
      params[0:5]   = c_i  (domain biases)
      params[5:10]  = k_i  (domain scales)
      params[10:15] = t_j  (global mixture sensitivities)
    
    Args:
        proportions: Array [n_samples, 5], each row sums to 1.0
        params:      Array of length ≥15 (will be padded/truncated to 15)
        
    Returns:
        loss_predictions: Array [n_samples, 5]
    """
    # Ensure 2D
    proportions = np.atleast_2d(proportions)
    n_samples, n_domains = proportions.shape
    # Pad or truncate params to length 15
    if len(params) < 15:
        p = np.ones(15, dtype=float)
        p[:len(params)] = params
        params = p
    else:
        params = np.asarray(params[:15], dtype=float)
    # Unpack parameters
    c = params[0:5]       # domain biases
    k = params[5:10]      # domain scales
    t = params[10:15]     # global mixture sensitivities
    # Compute shared mixture score per sample
    # sum_j t_j * r_j
    mixture_score = np.dot(proportions, t)
    # exp of mixture
    exp_mix = np.exp(mixture_score)
    # Predict losses: broadcast exp_mix across domains
    # L[i] = c[i] + k[i] * exp_mix
    loss_predictions = c + np.outer(exp_mix, k)
    return loss_predictions

def fit_scaling_law(proportions, loss_values):
    """
    Fit the scaling law to domain proportions and loss values
    
    Args:
        proportions: Array of domain proportions [n_samples, 5]
        loss_values: Array of corresponding loss values [n_samples, 5]
        
    Returns:
        Optimized parameters (15 parameters: c_i, k_i, t_j)
    """
    # Initialize 15 parameters (5 c's, 5 k's, 5 t's)
    initial_params = np.ones(15)
    # Set c_i to mean loss per domain, k_i=1, t_j=1
    for i in range(5):
        initial_params[i]       = np.mean(loss_values[:, i])  # c_i
        initial_params[i + 5]   = 1.0                         # k_i
        initial_params[i + 10]  = 1.0                         # t_j
    
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
# EVOLVE-BLOCK-END