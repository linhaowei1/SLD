# EVOLVE-BLOCK-START
"""
Enhanced scaling law discovery for LLM finetuning scenarios
Features:
- Theoretically grounded power-law form with offset
- Log‐parameterization to enforce positivity
- Robust Huber loss to mitigate outliers
- Global (Differential Evolution) + local (L-BFGS-B) hybrid optimization
- Smart initialization via log‐linear regression on data
- Parameter bounds and numerical stability checks
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

def scaling_law_func(data_points, params):
    """
    Predict loss as a * (x + x0)^(-b) + c, with enforced positivity for a, b, x0.
    params: [log_a, b, x0, c]
      - a     = exp(log_a) > 0
      - b     = max(params[1], 0)
      - x0    = max(params[2], 0)
      - c     = params[3] (baseline offset)
    """
    x = np.asarray(data_points, dtype=float)
    # unpack and enforce constraints
    log_a, b_raw, x0_raw, c = params
    a = np.exp(log_a)
    b = max(b_raw, 0.0)
    x0 = max(x0_raw, 0.0)
    # compute prediction
    y = a * np.power(x + x0, -b) + c
    # guard against nan/inf
    y = np.where(np.isfinite(y), y, np.finfo(float).max)
    return y

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the scaling law to (data_points, loss_values) via a two‐stage optimizer:
      1) Differential Evolution for global search
      2) L-BFGS-B for local refinement
    Uses a robust Huber loss to reduce sensitivity to outliers.
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Smart initialization if none provided
    if initial_params is None:
        # baseline offset c0 ~ 90% of min observed loss
        c0 = max(np.min(y) * 0.9, 1e-6)
        # adjusted loss for power-law fit
        y_adj = y - c0
        y_adj = np.clip(y_adj, 1e-6, None)
        # fit log-log: log(y_adj) = log(a) - b*log(x)
        log_x = np.log(x + 1e-8)
        log_y = np.log(y_adj)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        b0 = max(-slope, 1e-3)
        log_a0 = intercept
        x0_0 = np.min(x) * 0.1
        initial_params = np.array([log_a0, b0, x0_0, c0], dtype=float)

    # Parameter bounds: [(log_a), (b), (x0), (c)]
    bounds = [(-20, 20),    # log_a
              (0.0, 10.0),  # b
              (0.0, np.max(x)),  # x0
              (0.0, np.max(y))]  # c

    # Robust Huber loss
    def huber_loss(res, delta):
        is_small = np.abs(res) <= delta
        loss = np.empty_like(res)
        loss[is_small] = 0.5 * res[is_small]**2
        loss[~is_small] = delta * (np.abs(res[~is_small]) - 0.5 * delta)
        return np.mean(loss)

    def objective(params):
        y_pred = scaling_law_func(x, params)
        if np.any(~np.isfinite(y_pred)):
            return 1e6
        res = y_pred - y
        # use delta = 1 * std(res) or 1e-3 if std is zero
        delta = max(np.std(res), 1e-3)
        return huber_loss(res, delta)

    # 1) Global search
    de_result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=8,
        tol=1e-3,
        polish=False,
        disp=False
    )

    # 2) Local refinement
    local_result = minimize(
        objective,
        de_result.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    best_params = local_result.x if local_result.success else de_result.x
    return best_params

# Expose expected parameter count
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END


import pandas as pd
import os

if __name__ == "__main__":
    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    # Common data sizes corresponding to columns (excluding the zero‐size column)
    data_sizes = np.array([200, 400, 800, 1600, 3200, 6400, 12800,
                           25600, 51200, 102400, 204800, 409600,
                           819200, 1638400])

    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {csv_file}")
        print(f"{'='*50}")

        df = pd.read_csv(os.path.join(data_dir, csv_file))
        loss_columns = df.columns[1:-2]  # skip model name & final metadata cols

        for _, row in df.iterrows():
            model_name = row['config name']
            # collect valid (size, loss) pairs, skipping the zero‐size
            xs, ys = [], []
            for idx, col in enumerate(loss_columns[1:], start=1):
                val = row[col]
                if pd.notna(val) and val > 0:
                    xs.append(data_sizes[idx-1])
                    ys.append(float(val))
            if len(xs) < 4:
                print(f"Model {model_name}: insufficient data points, skipping.")
                continue

            xs = np.array(xs)
            ys = np.array(ys)
            print(f"\nModel: {model_name}")
            print(f"Data points: {len(xs)}, Range: [{xs[0]}, {xs[-1]}]")
            print(f"Loss range: [{ys[-1]:.4f}, {ys[0]:.4f}]")

            # Fit scaling law
            fitted = fit_scaling_law(xs, ys)
            print(f"Fitted params: {fitted}")

            # Evaluate fit quality
            pred = scaling_law_func(xs, fitted)
            mse = np.mean((pred - ys)**2)
            print(f"MSE: {mse:.6f}")
            print(f"Model size: {row['size']:,} params | Family: {row['family']}")