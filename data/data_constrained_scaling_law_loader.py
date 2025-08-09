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
    Loads data-constrained scaling law data. All data is in a single group.
    - Features (X): [tokens, params, unique_tokens]
    - Labels (y): loss
    """
    np.random.seed(random_seed)
    df = pd.read_csv(data_dir / "data.csv")
    
    # Feature order is important
    X = df[['tokens', 'params', 'unique_tokens']].values
    y = df['loss'].values
    
    # leave largest params and tokens for test
    max_data_size = sorted(df["unique_tokens"].unique())[-2]
    max_param_size = sorted(df["params"].unique())[-2] 

    test_mask = (df["unique_tokens"] >= max_data_size) | (df["params"] >= max_param_size)
    
    if train:
        train_mask = ~test_mask
        return {"all_data": (X[train_mask], y[train_mask])}
    else:
        return {"all_data": (X[test_mask], y[test_mask])}
        