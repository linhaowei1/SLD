# EVOLVE-BLOCK-START
"""
Scaling law discovery for LLM finetuning scenarios
Enhanced with a robust power‐law form and multi‐start + hybrid optimization
"""
import numpy as np
from scipy.optimize import minimize

def scaling_law_func(data_points, params):
    """
    A robust power‐law scaling function:
        loss = a * (data + x0)^(-b) + c
    with enforced parameter bounds for stability.
    Args:
        data_points: array-like of training set sizes
        params: [a, b, x0, c]
    Returns:
        loss predictions
    """
    # Unpack and enforce positivity
    a, b, x0, c = params
    a = max(a, 1e-12)
    b = max(b, 1e-12)
    x0 = max(x0, 0.0)
    c = max(c, 0.0)

    x = np.asarray(data_points, dtype=float)
    # Offset and avoid zeros
    x_safe = x + x0
    x_safe = np.maximum(x_safe, 1e-12)

    # Compute power‐law
    loss = a * x_safe**(-b) + c
    return loss

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    Fit the scaling law via multi‐start L-BFGS-B and Nelder-Mead refinement.
    Args:
        data_points: array-like sizes
        loss_values: array-like measured losses
        initial_params: optional initial guess (unused in multi-start)
    Returns:
        best-fit [a, b, x0, c]
    """
    x = np.asarray(data_points, dtype=float)
    y = np.asarray(loss_values, dtype=float)

    # Objective: MSE
    def obj(p):
        pred = scaling_law_func(x, p)
        return np.mean((pred - y)**2)

    # Parameter bounds: a,b>=1e-8; x0,c>=0
    bounds = [(1e-8, None), (1e-8, None), (0.0, None), (0.0, None)]

    # Multi-start random inits
    best = {'fun': np.inf, 'x': None}
    rng = np.random.RandomState(0)
    y_min, y_max = y.min(), y.max()
    d_min = x.min()

    for _ in range(8):
        # Smart initial guesses
        a0 = max(y_max - y_min, 1e-2) * rng.uniform(0.5, 2.0)
        b0 = rng.uniform(0.1, 2.0)
        x00 = rng.uniform(0.0, d_min)
        c0 = max(y_min, 1e-3) * rng.uniform(0.0, 1.0)
        init = np.array([a0, b0, x00, c0])

        try:
            res = minimize(obj, init, method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-9, 'maxiter': 5000})
            if res.fun < best['fun']:
                best = {'fun': res.fun, 'x': res.x}
        except Exception:
            continue

    # Fallback if all starts fail
    if best['x'] is None:
        best['x'] = np.array([1.0, 1.0, 0.0, 0.0])
        best['fun'] = obj(best['x'])

    # Nelder-Mead refinement
    try:
        res_nm = minimize(obj, best['x'], method='Nelder-Mead',
                          options={'maxiter': 2000, 'fatol': 1e-9})
        if res_nm.fun < best['fun']:
            best = {'fun': res_nm.fun, 'x': res_nm.x}
    except Exception:
        pass

    return best['x']

# Inform how many params we expect
scaling_law_func.num_params = 4
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import os
    import pandas as pd

    data_dir = "data"
    csv_files = ["flan.csv", "gigaword.csv", "wmt19.csv"]
    # Fixed data‐size grid
    data_sizes = np.array([
        200, 400, 800, 1600, 3200, 6400, 12800,
        25600, 51200, 102400, 204800, 409600,
        819200, 1638400
    ], dtype=float)

    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"处理数据集: {csv_file}")
        print(f"{'='*50}")
        df = pd.read_csv(os.path.join(data_dir, csv_file))

        # Assume first column is 'config name', last two are 'size' & 'family'
        loss_cols = df.columns[1:-2]

        for _, row in df.iterrows():
            model_name = row['config name']
            sizes, losses = [], []
            # Skip zero‐size column, align with data_sizes
            for idx, col in enumerate(loss_cols[1:], start=1):
                val = row[col]
                if pd.notna(val) and val > 0:
                    sizes.append(data_sizes[idx-1])
                    losses.append(float(val))

            if len(sizes) < 4:
                print(f"\n模型 {model_name}: 数据点不足，跳过拟合")
                continue

            sizes = np.array(sizes, dtype=float)
            losses = np.array(losses, dtype=float)

            print(f"\n模型: {model_name}")
            print(f"数据点数量: {len(sizes)}")
            print(f"数据大小范围: {int(sizes[0])} – {int(sizes[-1])}")
            print(f"损失值范围: {losses[-1]:.3f} – {losses[0]:.3f}")

            # Fit scaling law
            params = fit_scaling_law(sizes, losses)
            preds = scaling_law_func(sizes, params)
            mse = np.mean((preds - losses)**2)

            print(f"拟合参数: {params}")
            print(f"均方误差: {mse:.6f}")
            print(f"模型大小: {int(row['size']):,} 参数")
            print(f"模型家族: {row['family']}")