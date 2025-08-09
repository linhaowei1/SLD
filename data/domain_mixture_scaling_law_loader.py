import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple

def load_data_for_task(
    data_dir: Path,
    train: bool = True,
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads domain mixture scaling law data.
    Each model size is a separate group.
    - Features (X): proportions (5 domains)
    - Labels (y): loss (5 domains)
    """
    np.random.seed(random_seed)
    with open(data_dir / "all_models_data.json", 'r') as f:
        data = json.load(f)

    grouped_data = {}
    split_key = "train" if train else "eval"

    for model_size, model_data in data.items():
        if model_size == "1B": continue

        points = model_data.get(split_key, [])
        if not points: continue
        
        proportions = np.array([p['proportions'] for p in points])
        losses = np.array([p['loss'] for p in points])
        
        # Here, X has shape (n_mixtures, 5) and y has shape (n_mixtures, 5)
        # This is a special case where the response is also multi-dimensional.
        grouped_data[model_size] = (proportions, losses)
            
    return grouped_data