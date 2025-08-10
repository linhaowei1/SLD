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
    Loads learning rate scaling law data.
    Test set consists of points with the largest data_size or non-embedding_param_size.
    - Features (X): [lr, bsz, data_size, non-embedding_param_size]
    - Labels (y): lm loss
    """
    df = pd.read_csv(data_dir / "dense_lr_bs_loss.csv")
    df = df[df['smooth loss'] <= 4.0].copy() # Filter outliers
    
    df.rename(columns={'non-embedding_param_size': 'non_embedding_param_size'}, inplace=True)
    
    # Feature order is important
    feature_cols = ['lr', 'bs', 'D', 'N']
    X = df[feature_cols].values
    y = df['smooth loss'].values

    max_data_size = df["D"].max()
    max_param_size = df["N"].max()

    test_mask = (df["D"] == max_data_size) & (df["N"] == max_param_size)
    
    if train:
        train_mask = ~test_mask
        return {"all_data": (X[train_mask], y[train_mask])}
    else:
        return {"all_data": (X[test_mask], y[test_mask])}