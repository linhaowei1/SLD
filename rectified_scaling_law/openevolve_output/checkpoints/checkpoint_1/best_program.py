# EVOLVE-BLOCK-START
"""
Evolved scaling law discovery for LLM finetuning scenarios
Improved mathematical form, robust multi-start fitting, and clearer code structure.
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    Rational-form scaling law with 4 parameters:
        loss = a / (x^c + b) + d

    Args:
        data_points: array-like of training data sizes
        params: array-like of 4 parameters [a, b, c, d]
    Returns:
        numpy array of predicted losses
    """
    # Enforce exactly 4 parameters
    p = np.asarray(params, dtype=float).flatten()
    if p.size != 4:
        p = np.resize(p, 4)
    a, b, c, d = p

    # Ensure numerical stability
    x = np.asarray(data_points, dtype=float)
    x_safe = np.maximum(x, 1e-8)

    # Compute rational scaling law
    # a > 0 controls scale, b > 0 is denom offset, c > 0 exponent, d >= 0 baseline
    return a / (np.power(x_safe, c) + b) + d

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the 4-parameter scaling law to empirical data using multi-start L-BFGS-B.

    Args:
        data_points: array-like of training data sizes
        loss_values: array-like of observed losses
        initial_params: optional 4-element initial guess
    Returns:
        best-fit parameters as numpy array of length 4
    """
    # Prepare data
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)
    # Filter valid entries
    mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0)
    x, y = x[mask], y[mask]
    if x.size < 4:
        # Not enough data, return default
        return np.array([1.0, 1.0, 1.0, float(np.min(y) if y.size else 0.0)])

    # Define objective: mean squared error
    def obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y) ** 2)

    # Parameter bounds
    y_max, y_min = np.max(y), np.min(y)
    bounds = [
        (1e-6, (y_max - y_min) * (x.min() ** 1 + 1) * 10 + 1e-6),  # a
        (1e-6, np.max(x) ** 1 + 1e6),                              # b
        (1e-3, 5.0),                                               # c
        (0.0, y_max)                                               # d
    ]

    # Generate multi-start initial guesses
    inits = []
    # 1) User-provided
    if initial_params is not None:
        p0 = np.asarray(initial_params, dtype=float).flatten()
        if p0.size == 4:
            inits.append(p0)
    # 2) Heuristic guess based on data
    b0, c0 = 1.0, 1.0
    d0 = y_min
    a0 = max((y_max - d0) * (x.min() ** c0 + b0), 1e-6)
    inits.append(np.array([a0, b0, c0, d0]))
    # 3) Midpoint of bounds
    mid = np.array([(lo + hi) / 2 for lo, hi in bounds])
    inits.append(mid)
    # 4) Small variations around mid
    inits.append(mid * np.array([0.5, 2.0, 1.0, 1.0]))
    # 5) Random in bounds
    rng = np.random.RandomState(42)
    rand_init = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
    inits.append(rand_init)

    best_p, best_obj = None, np.inf
    # Run L-BFGS-B from each start
    for p0 in inits:
        try:
            res = minimize(
                obj,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol':1e-9, 'gtol':1e-5, 'maxiter':1000}
            )
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_p = res.x
        except Exception:
            continue

    # Fallback if no successful fit
    if best_p is None:
        best_p = inits[1]

    # Ensure length 4
    best_p = np.asarray(best_p, dtype=float).flatten()
    if best_p.size != 4:
        best_p = np.resize(best_p, 4)
    return best_p

# Informational attribute
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    data_sizes = np.array([200, 400, 800, 1600, 3200, 6400,
                           12800, 25600, 51200, 102400,
                           204800, 409600, 819200, 1638400])

    for csv_file in csv_files:
        print(f"\n{'='*50}\nProcessing dataset: {csv_file}\n{'='*50}")
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        # Loss columns exclude ['config name', 'size', 'family']
        loss_cols = [c for c in df.columns if c not in ('config name','size','family')]

        for _, row in df.iterrows():
            model_name = row['config name']
            # Collect (size, loss) pairs skipping zero or NaN
            sizes, losses = [], []
            for i, col in enumerate(loss_cols):
                x = data_sizes[i] if i < len(data_sizes) else None
                y = row[col]
                if x and pd.notna(y) and y > 0:
                    sizes.append(x)
                    losses.append(float(y))
            if len(sizes) < 4:
                print(f"Model {model_name}: insufficient data, skipping.")
                continue

            sizes = np.array(sizes)
            losses = np.array(losses)
            print(f"\nModel: {model_name}")
            print(f"Data sizes: {sizes.min()} - {sizes.max()} ({len(sizes)} points)")
            print(f"Losses: {losses.max():.4f} -> {losses.min():.4f}")

            # Fit and evaluate
            params = fit_scaling_law(sizes, losses)
            preds = scaling_law_func(sizes, params)
            mse = np.mean((preds - losses) ** 2)

            print(f"Fitted params: {params}")
            print(f"Mean squared error: {mse:.6f}")
            print(f"Model size: {int(row['size']):,} parameters")
            print(f"Model family: {row['family']}")