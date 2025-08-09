import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

def load_data_for_task(
    data_dir: Path,
    train: bool = True,
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads MoE scaling law data. All data is in a single group.
    - Features (X): [num_experts, dense_parameter_count]
    - Labels (y): loss_validation
    """
    np.random.seed(random_seed)
    df = pd.read_csv(data_dir / "data.csv")
    df = df[df['step'] == 249000].copy()

    # Feature order is important
    X = df[['num_experts', 'dense_parameter_count']].values
    y = df['loss_validation'].values

    # Shuffle and split
    max_param_size = df["dense_parameter_count"].max()

    test_mask = (df["dense_parameter_count"] == max_param_size)

    if train:
        train_mask = ~test_mask
        return {"all_data": (X[train_mask], y[train_mask])}
    else:
        return {"all_data": (X[test_mask], y[test_mask])}