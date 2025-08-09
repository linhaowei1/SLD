import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Tuple

def load_data_for_task(
    data_dir: Path,
    train: bool = True,
    random_seed: int = 42
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads rectified scaling law data.
    Each (model, dataset) combination is a separate group.
    - Features (X): data_size
    - Labels (y): loss_values
    """
    np.random.seed(random_seed)
    grouped_data = {}
    csv_files = ["flan.csv", "gigaword.csv", "wikiword.csv"]
    
    for csv_file in csv_files:
        df = pd.read_csv(data_dir / csv_file)
        dataset_name = csv_file.replace('.csv', '')
        loss_columns = df.columns[2:]
        data_sizes = np.array([int(c) for c in loss_columns])

        for _, row in df.iterrows():
            model_name = row[df.columns[0]]
            loss_values = pd.to_numeric(row[loss_columns], errors='coerce').values
            
            valid_mask = ~np.isnan(loss_values) & (loss_values > 0)
            valid_sizes = data_sizes[valid_mask]
            valid_losses = loss_values[valid_mask]

            # Test set is the point with the largest data size (819200)
            test_size = 819200
            is_test_point = valid_sizes == test_size
            
            if train:
                train_mask = ~is_test_point
                if np.sum(train_mask) >= 4: # Need enough points to fit
                    X = valid_sizes[train_mask].reshape(-1, 1)
                    y = valid_losses[train_mask]
                    grouped_data[(model_name, dataset_name)] = (X, y)
            else: # Test data
                if np.any(is_test_point):
                    X = valid_sizes[is_test_point].reshape(-1, 1)
                    y = valid_losses[is_test_point]
                    grouped_data[(model_name, dataset_name)] = (X, y)
                    
    return grouped_data