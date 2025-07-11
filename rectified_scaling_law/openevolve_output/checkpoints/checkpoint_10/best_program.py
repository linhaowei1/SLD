# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Simplified, robust 4-parameter power-law model with multi-start L-BFGS-B fitting.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    4-parameter scaling law: loss = d + a * (x + c)^(-b)
    Params:
        a (scale), b (exponent), c (offset), d (baseline)
    """
    # Unpack exactly 4 parameters
    a, b, c, d = params
    x = np.asarray(data_points, dtype=float)
    # Ensure x+c > 0
    xpc = x + np.abs(c)
    # Compute prediction
    return np.clip(d + a * np.power(xpc, -b), 0.0, None)

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the 4-parameter scaling law via multi-start L-BFGS-B.
    Returns an array of 4 optimized parameters.
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Heuristic initialization if not provided
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    d0 = max(0.0, y_min * 0.5)      # baseline ~ half of min loss
    b0 = 0.5                        # initial exponent
    c0 = max(0.0, x_min * 0.01)     # small offset
    a0 = max(1e-6, (y_max - d0) * (x_min + c0)**b0)
    if initial_params is None:
        init = np.array([a0, b0, c0, d0], dtype=float)
    else:
        arr = np.asarray(initial_params, dtype=float)
        if arr.size != 4:
            raise ValueError("initial_params must be length 4")
        init = arr.copy()

    # Bounds: ensure positivity where needed
    bounds = [
        (1e-8, None),  # a >= 0
        (1e-8, None),  # b >= 0
        (0.0, None),   # c >= 0
        (0.0, None)    # d >= 0
    ]

    def mse_obj(p):
        pred = scaling_law_func(x, p)
        err = pred - y
        return np.mean(err * err)

    # Multi-start optimization
    best_params = None
    best_loss = np.inf
    # generate 5 starts: the heuristic + random perturbations
    starts = [init] + [
        init * np.random.uniform(0.5, 1.5, size=4) for _ in range(4)
    ]
    for start in starts:
        res = minimize(mse_obj, start, method='L-BFGS-B', bounds=bounds)
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    # Fallback to initial if optimization failed
    if best_params is None:
        best_params = init

    return best_params

# Declare expected param count
scaling_law_func.num_params = 4

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    # Shared data sizes
    data_sizes = np.array(
        [200, 400, 800, 1600, 3200, 6400, 12800,
         25600, 51200, 102400, 204800, 409600, 819200, 1638400]
    )

    for fname in csv_files:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        print(f"\n=== Dataset: {fname} ===")
        # Assume 'config name', 'size', 'family' are columns
        loss_cols = [c for c in df.columns if c not in ('config name', 'size', 'family')]
        for _, row in df.iterrows():
            model = row['config name']
            # collect valid losses (skip first column if zero-size)
            ys, xs = [], []
            for i, col in enumerate(loss_cols):
                val = row[col]
                if pd.notna(val) and val > 0:
                    ys.append(float(val))
                    xs.append(data_sizes[i])
            ys = np.array(ys)
            xs = np.array(xs)
            if xs.size < 4:
                print(f"Model {model}: insufficient data, skipping.")
                continue

            # Fit and evaluate
            params = fit_scaling_law(xs, ys)
            pred = scaling_law_func(xs, params)
            mse = np.mean((pred - ys) ** 2)

            print(f"\nModel: {model}")
            print(f"Data sizes: {xs.min()}–{xs.max()} ({xs.size} points)")
            print(f"Loss range: {ys.max():.4f}→{ys.min():.4f}")
            print(f"Fitted params: {[float(f'{p:.6g}') for p in params]}")
            print(f"MSE: {mse:.6f}")
            print(f"Params: size={int(row['size']):,}, family={row['family']}")