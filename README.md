# EvoSLD — Evolutionary Scaling Law Discovery

*EvoSLD* uses evolutionary computation to **discover and optimize scaling laws** across machine‑learning scenarios. It’s built on top of the [OpenEvolve](https://github.com/codelion/openevolve) framework and evolves **both** the functional form of scaling laws **and** their fitting algorithms.

---

## Highlights

* **End‑to‑end discovery**: Evolves closed‑form scaling functions *and* bespoke optimizers.
* **Multiple domains** out of the box:

  * **Data‑Constrained**: how training data characteristics relate to loss
  * **Domain Mixture**: effect of mixing domains on performance
  * **Learning Rate**: scaling with learning rate & batch size
  * **Mixture of Experts (MoE)**: scaling behavior in MoE architectures
  * **Rectified (SFT)**: rectified scaling laws for supervised fine‑tuning
  * **Vocabulary**: impact of vocabulary size
* **Constraint‑aware**: hard limit of **≤ 7 parameters** in discovered functions for tractable fitting.
* **Robust evaluation**: timeouts, retries, cross‑validation hooks, and checkpointed evolution.
* **Batchable & reproducible**: run many tasks and seeds; deterministic seeding throughout.

---

## Requirements

* **Python 3.13+**
* **[uv](https://docs.astral.sh/uv/)** package & project manager
* An **OpenAI‑compatible** API key (set `OPENAI_API_KEY`)

> **Note**: `uv run` guarantees commands execute inside a synchronized project environment. If you prefer plain `pip`, you can adapt the commands accordingly.

---

## Quick Start (with `uv`)

```bash
# 1) Clone
git clone <repository-url>
cd evosld

# 2) Install project dependencies
uv sync

# 3) Provide LLM access (OpenAI‑compatible endpoint)
export OPENAI_API_KEY=your_key

# 4) Run a single discovery (example: Data‑Constrained)
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run openevolve-run.py \
  --config configs/data_constrained_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/data_constrained_scaling_law/run_1

# 5) Run all tasks (batch execution)
# If your script is executable (shebang present), this works:
uv run scripts/run.sh
# or
bash scripts/run.sh
```

---

## Project Layout

```
evosld/
├─ configs/                     # YAML configs (one per scaling law)
│  ├─ data_constrained_scaling_law.yaml
│  ├─ domain_mixture_scaling_law.yaml
│  ├─ lr_scaling_law.yaml
│  ├─ moe_scaling_law.yaml
│  ├─ rectified_scaling_law.yaml
│  └─ vocab_scaling_law.yaml
├─ data/                        # Data files & loaders
│  ├─ {task_name}/              # One folder per task
│  │  ├─ data.csv               # CSV with features + target
│  │  └─ {task_name}_loader.py  # Task‑specific loader
│  └─ ...
├─ data_loader.py               # Unified data loading interface
├─ evaluator.py                 # Unified evaluation system
├─ init_program.py              # Initial scaling-law template
├─ results/                     # Outputs & checkpoints
└─ scripts/
   └─ run.sh                    # Batch execution helper
```

---

## Running Existing Tasks

### Single Task

```bash
export EVAL_TASK_NAME="data_constrained_scaling_law"
uv run python openevolve-run.py \
  --config configs/data_constrained_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/data_constrained_scaling_law/run_1
```

### Batch Mode

```bash
bash scripts/run.sh
```

This will:

* Run each task **3 times** with different random seeds
* Write to `results/{task_name}/run_{1,2,3}/`
* Save intermediate **checkpoints**
* Evaluate and materialize the **best program** per run

### Evaluate a Discovered Program

```bash
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run python evaluator.py \
  results/data_constrained_scaling_law/run_1/best/best_program.py
```

---

## How to Add a New Scaling Law

### 1) Create a Config

Create `configs/your_law_name.yaml`:

````yaml
# Configuration for your scaling law discovery
max_iterations: 50
checkpoint_interval: 1
log_level: "INFO"
random_seed: 42

# LLM configuration
llm:
  models:
    - name: "o4-mini"
      weight: 1.0
  api_base: "http://api.llm.wq/v1"  # Any OpenAI‑compatible endpoint
  max_tokens: 16384
  timeout: 120
  retries: 3
  retry_delay: 5

# Prompt configuration
prompt:
  system_message: |
    You are an expert in scaling laws who specializes in discovering scaling law functions
    for [describe your domain]. Your task is to evolve both the `scaling_law_func` function
    and the `fit_scaling_law` optimization algorithm.

    **IMPORTANT: The scaling law function must use no more than 7 parameters.**

    **DATA CHARACTERISTICS:**
    - Features: [describe your features] - [N]D input
    - Labels: [describe your target] - scalar output
    - [Describe the data relationships you want to model]

    The function signatures must remain:
    ```python
    def scaling_law_func(data_points, params):
        # data_points: (N,F) array with your features
        # params: Array of up to 7 parameters
        # Returns: Predicted values

    def fit_scaling_law(data_points, target_values):
        # data_points: (N,F) array with your features  
        # target_values: Array of corresponding targets
        # Returns: Optimized parameters (up to 7 parameters)
    ```

  num_top_programs: 3
  num_diverse_programs: 2
  use_template_stochasticity: true

# Database / evolution configuration
database:
  population_size: 100
  archive_size: 50
  num_islands: 3
  migration_interval: 20
  migration_rate: 0.1
  elite_selection_ratio: 0.1
  exploration_ratio: 0.2
  exploitation_ratio: 0.7
  feature_dimensions: ["combined_score"]
  feature_bins: 10

# Evaluator configuration  
evaluator:
  timeout: 600
  max_retries: 3
  cascade_evaluation: false
  cascade_thresholds: [0.3, 0.6]
  parallel_evaluations: 4
  use_llm_feedback: false

# Evolution settings
diff_based_evolution: false
max_code_length: 10000
````

### 2) Prepare Data

```bash
mkdir -p data/your_law_name
```

Add `data/your_law_name/data.csv` with columns like:

```csv
feature1,feature2,feature3,target
1.0,2.0,3.0,0.5
2.0,4.0,6.0,0.3
...
```

### 3) Create a Data Loader

`data/your_law_name_loader.py`:

```python
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
    Loads data for your scaling law task.
    - Features (X): [feature1, feature2, feature3, ...]
    - Labels (y): target
    """
    np.random.seed(random_seed)
    df = pd.read_csv(data_dir / "data.csv")

    feature_columns = ['feature1', 'feature2', 'feature3']
    X = df[feature_columns].values
    y = df['target'].values

    # 80/20 split with shuffling
    n = len(df)
    n_train = int(0.8 * n)
    idx = np.random.permutation(n)
    indices = idx[:n_train] if train else idx[n_train:]

    return {"all_data": (X[indices], y[indices])}
```

### 4) Register the Task

Add your task to `TASK_CONFIG` in `evaluator.py`:

```python
TASK_CONFIG = {
    # ... existing tasks ...
    "your_law_name": {
        "scaling_vars": ["feature1", "feature2", "feature3"],
        "response_var": "target",
    },
}
```

### 5) (Optional) Batch Script

Append to `scripts/run.sh`:

```bash
tasks=(
  # ... existing tasks ...
  "your_law_name"
)
```

---

## Configuration Tips

* **Search budget**: increase `max_iterations` and `population_size` to explore more aggressively.
* **Exploration vs. exploitation**: tune `exploration_ratio`/`exploitation_ratio`.
* **Parallelism**: raise `parallel_evaluations` to speed up evaluation throughput.
* **Reproducibility**: keep `random_seed` fixed for apples‑to‑apples comparisons.
* **API resilience**: bump `llm.timeout` and `llm.retries` for flaky networks/providers.

---

## Results & Checkpoints

After runs, you’ll find:

```
results/{task_name}/{run_id}/
├─ checkpoints/
│  ├─ checkpoint_1/
│  │  ├─ best_program.py         # Best program at this checkpoint
│  │  ├─ best_program_info.json
│  │  ├─ metadata.json
│  │  └─ programs/               # All candidate programs
│  └─ ...
├─ logs/                         # Execution logs
└─ best/                         # Final best program (symlink)
   └─ best_program.py
```

---

## Troubleshooting

* **Import errors** → re‑sync deps: `uv sync`
* **Task not found** → ensure `EVAL_TASK_NAME` matches a key in `TASK_CONFIG`
* **Data loader shape issues** → return `Dict[str, Tuple[np.ndarray, np.ndarray]]`
* **Evolution stalls** → increase `exploration_ratio`, population size, or iterations
* **API timeouts** → increase `llm.timeout` / `llm.retries`; check your `api_base`
* **Batch script not executable** → `chmod +x scripts/run.sh` or run with `bash scripts/run.sh`

---

## FAQ

**Q: Do I have to use OpenAI specifically?**
A: No. Any **OpenAI‑compatible** endpoint works (set `api_base` in the YAML).

**Q: Can I use plain `pip` instead of `uv`?**
A: Yes—create/activate a virtualenv and install requirements; then run the same commands using `python`.

---

## Acknowledgments

* Built on the excellent **[OpenEvolve](https://github.com/codelion/openevolve)** evolutionary coding framework.

---

## Citation

If you use EvoSLD in academic work, please cite this repository and OpenEvolve.
