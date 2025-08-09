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
    Loads vocab scaling law data. All data is in a single group.
    - Features (X): [Non_vocab_parameters, vocab_size, num_characters]
    - Labels (y): Lossu
    """
    np.random.seed(random_seed)
    df = pd.read_csv(data_dir / "data.csv")

    # Feature order is important
    X = df[['Non_vocab_parameters', 'vocab_size', 'num_characters']].values
    y = df['Lossu'].values

    # Shuffle and split
    max_param_size = df["vocab_size"].max()

    test_mask = (df["vocab_size"] == max_param_size)

    if train:
        train_mask = ~test_mask
        return {"all_data": (X[train_mask], y[train_mask])}
    else:
        return {"all_data": (X[test_mask], y[test_mask])}