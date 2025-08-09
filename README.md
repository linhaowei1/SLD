# EvoSLD 🧬 — Evolutionary Scaling Law Discovery

**EvoSLD** uses evolutionary computation to **discover and optimize scaling laws** 📈 across machine-learning scenarios. It sits on top of the [OpenEvolve](https://github.com/codelion/openevolve) framework and co-evolves both the **functional form** of scaling laws **and** their **fitting algorithms** 🤖.

> 📄 **Paper**: [EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models](https://arxiv.org/abs/2507.21184)

-----

## 📖 Table of Contents

  * [Why EvoSLD?](https://www.google.com/search?q=%23%F0%9F%A4%94-why-evosld)
  * [Features](https://www.google.com/search?q=%23%E2%9C%A8-features)
  * [What’s included (tasks)](https://www.google.com/search?q=%23%F0%9F%93%A6-whats-included-tasks)
  * [Requirements](https://www.google.com/search?q=%23%F0%9F%93%8B-requirements)
  * [Install](https://www.google.com/search?q=%23%F0%9F%9B%A0%EF%B8%8F-install)
      * [Using `uv` (recommended)](https://www.google.com/search?q=%23using-uv-recommended)
      * [Using plain `pip`](https://www.google.com/search?q=%23using-plain-pip)
  * [Quick Start](https://www.google.com/search?q=%23%F0%9F%9A%80-quick-start)
  * [Project Layout](https://www.google.com/search?q=%23%F0%9F%93%82-project-layout)
  * [Running Tasks](https://www.google.com/search?q=%23-running-tasks)
  * [Evaluating a Discovered Program](https://www.google.com/search?q=%23%F0%9F%93%8A-evaluating-a-discovered-program)
  * [Add a New Scaling Law](https://www.google.com/search?q=%23%E2%9E%95-add-a-new-scaling-law)
  * [Configuration Guide](https://www.google.com/search?q=%23%E2%9A%99%EF%B8%8F-configuration-guide)
  * [Data Interface](https://www.google.com/search?q=%23-data-interface)
  * [Tips for Search/Evolution](https://www.google.com/search?q=%23%F0%9F%92%A1-tips-for-searchevolution)
  * [Troubleshooting](https://www.google.com/search?q=%23%F0%9F%86%98-troubleshooting)
  * [FAQ](https://www.google.com/search?q=%23%E2%9D%93-faq)
  * [Cite](https://www.google.com/search?q=%23%E2%9C%8D%EF%B8%8F-cite)
  * [Acknowledgments](https://www.google.com/search?q=%23%F0%9F%99%8F-acknowledgments)

-----

## 🤔 Why EvoSLD?

Scaling laws relate performance to factors like model size, dataset size, compute, and architecture. Hand-deriving such laws is time-consuming and often brittle. **EvoSLD** automates this by:

  * **Searching** 🧠 for symbolic forms of scaling laws (closed-form functions).
  * **Co-designing** 🧑‍🎨 the corresponding **fitting/optimization routine**.
  * **Selecting** ✅ candidates via evolutionary pressure on held-out data.

The result is a practical engine that can **rediscover** known laws and **propose better ones**—with explicit code you can inspect and re-use.

-----

## ✨ Features

  * **End-to-end discovery**: Evolves closed-form scaling functions *and* their bespoke optimizers.
  * **Multiple domains** out of the box:
      * **Data-Constrained**: How training data affects loss.
      * **Domain Mixture**: Effect of mixing domains on performance.
      * **Learning Rate**: Scaling with learning rate & batch size.
      * **Mixture of Experts (MoE)**: Behavior in MoE architectures.
      * **Rectified (SFT)**: Laws for supervised fine-tuning.
      * **Vocabulary**: Impact of vocabulary size.
  * **Customizable**: All stages (prompting, evolution, evaluation, data) are configurable.
  * **Checkpoints & reproducibility**: Periodic snapshots + seeds for reliable runs.

-----

## 📦 What’s included (tasks)

| Task key | Config file | Data folder |
| :--- | :--- | :--- |
| `data_constrained_scaling_law` | `configs/data_constrained_scaling_law.yaml` | `data/data_constrained_scaling_law/` |
| `domain_mixture_scaling_law` | `configs/domain_mixture_scaling_law.yaml` | `data/domain_mixture_scaling_law/` |
| `lr_scaling_law` | `configs/lr_scaling_law.yaml` | `data/lr_scaling_law/` |
| `moe_scaling_law` | `configs/moe_scaling_law.yaml` | `data/moe_scaling_law/` |
| `rectified_scaling_law` | `configs/rectified_scaling_law.yaml` | `data/rectified_scaling_law/` |
| `vocab_scaling_law` | `configs/vocab_scaling_law.yaml` | `data/vocab_scaling_law/` |

> Add your own tasks in the same pattern; see [Add a New Scaling Law](https://www.google.com/search?q=%23%E2%9E%95-add-a-new-scaling-law).

-----

## 📋 Requirements

  * **Python 3.13+**
  * **[`uv`](https://www.google.com/search?q=%5Bhttps://docs.astral.sh/uv/%5D\(https://docs.astral.sh/uv/\))** package manager (recommended)
  * An **OpenAI-compatible** API key (set `OPENAI_API_KEY`)
  * macOS/Linux/Windows

> **Note**: `uv run` guarantees commands execute inside a synchronized project environment. If you prefer plain `pip`, you can adapt the commands accordingly.

-----

## 🛠️ Install

### Using `uv` (recommended)

```bash
# 1) Clone the repo
git clone <repository-url>
cd evosld

# 2) Install dependencies
uv sync

# 3) Provide your LLM API key
export OPENAI_API_KEY=your_key
# Optional: if using a non-default endpoint
# export OPENAI_BASE_URL=https://your.openai.compatible.endpoint/v1
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="your_key"
# $env:OPENAI_BASE_URL="https://your.openai.compatible.endpoint/v1"
```

### Using plain `pip`

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt # Or use pyproject.toml

# Set your API key
export OPENAI_API_KEY=your_key
# export OPENAI_BASE_URL=https://your.openai.compatible.endpoint/v1
```

-----

## 🚀 Quick Start

Run a single discovery task (e.g., **Data-Constrained**):

```bash
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run openevolve-run.py \
  --config configs/data_constrained_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/data_constrained_scaling_law/run_1
```

Or run all tasks in batch:

```bash
# If the script is executable:
uv run scripts/run.sh
# Otherwise:
bash scripts/run.sh
```

-----

## 📂 Project Layout

```
evosld/
├─ configs/               # ⚙️ YAML configs (one per scaling law)
│  ├─ data_constrained_scaling_law.yaml
│  └─ ...
├─ data/                  # 📊 Data files & loaders
│  ├─ {task_name}/        # 📁 One folder per task
│  │  ├─ data.csv
│  │  └─ {task_name}_loader.py
│  └─ ...
├─ data_loader.py         # ↔️ Unified data loading interface
├─ evaluator.py           # ✅ Unified evaluation system
├─ init_program.py        # 🌱 Initial scaling-law template
├─ results/               # 🏆 Outputs & checkpoints
└─ scripts/
   └─ run.sh              # 🏃 Batch execution helper
```

-----

## 🏃 Running Tasks

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

  * Run each task **3 times** with different random seeds.
  * Write outputs to `results/{task_name}/run_{1,2,3}/`.
  * Save intermediate **checkpoints**.
  * Evaluate and save the **best program** from each run.

-----

## 📊 Evaluating a Discovered Program

```bash
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run python evaluator.py \
  results/data_constrained_scaling_law/run_1/best/best_program.py
```

-----

## ➕ Add a New Scaling Law

### 1\) Create a Config

Create `configs/your_law_name.yaml` and customize the settings (see the full template in the original README). Key sections include `llm`, `prompt`, `database`, and `evaluator`.

### 2\) Prepare Data

Create a directory for your data and add a `data.csv` file:

```bash
mkdir -p data/your_law_name
```

Your `data.csv` should have columns for features and the target variable.

### 3\) Create a Data Loader

Add a Python script `data/your_law_name_loader.py` to load your data. It must contain a `load_data_for_task` function that returns a dictionary containing NumPy arrays for features (X) and labels (y).

### 4\) Register the Task

Add your task to the `TASK_CONFIG` dictionary in `evaluator.py`:

```python
TASK_CONFIG = {
    # ... existing tasks ...
    "your_law_name": {
        "scaling_vars": ["your_feature1", "your_feature2"],
        "response_var": "your_target",
    },
}
```

### 5\) (Optional) Add to Batch Script

Add `"your_law_name"` to the `tasks` array in `scripts/run.sh` to include it in batch runs.

-----

## ⚙️ Configuration Guide

Key knobs to tune in your `.yaml` files:

  * **Search Budget**: Increase `max_iterations` and `population_size` for more thorough exploration.
  * **Exploration vs. Exploitation**: Adjust `exploration_ratio` and `exploitation_ratio`.
  * **Parallelism**: Raise `parallel_evaluations` to speed things up.
  * **Reproducibility**: Set a fixed `random_seed` for consistent results.
  * **API Resilience**: Bump `llm.timeout` and `llm.retries` for flaky networks.

-----

## ↔️ Data Interface

  * Your data loader should return a dictionary where values are tuples of `(features, target)`.
  * Features should be a 2D `(N, F)` NumPy array, and the target should be a 1D NumPy array.
  * The unified `data_loader.py` will use your task-specific loader based on the `EVAL_TASK_NAME`.

-----

## 💡 Tips for Search/Evolution

  * Start with a modest budget and inspect intermediate checkpoints in `results/`.
  * If evolution stalls, try increasing `population_size` or the `exploration_ratio`.
  * Consider grouping data into different regimes (e.g., compute-limited vs. data-limited) and evaluating on each subset for more nuanced insights.

-----

## 🆘 Troubleshooting

  * **Import Errors**: Run `uv sync` to ensure your environment is up-to-date.
  * **Task Not Found**: Check that `EVAL_TASK_NAME` matches a key in `TASK_CONFIG` in `evaluator.py`.
  * **API Timeouts**: Increase `llm.timeout` and `llm.retries` in your config, or check your `OPENAI_BASE_URL`.
  * **Script Not Executable**: Run `chmod +x scripts/run.sh` or execute it with `bash scripts/run.sh`.

-----

## ❓ FAQ

**Do I have to use OpenAI?**
No. Any OpenAI-compatible endpoint works. Just set the `api_base` in your YAML config or the `OPENAI_BASE_URL` environment variable.

**Can I use `pip` instead of `uv`?**
Yes. Create a virtual environment, activate it, and install dependencies from `requirements.txt`. Then run the Python commands directly.

**Where are the results stored?**
Under `results/{task_name}/{run_id}/`. You'll find checkpoints, logs, and the final `best/best_program.py`.

-----

## ✍️ Cite

If you use EvoSLD in your academic work, please cite the paper:

```bibtex
@article{lin2025evosld,
  title   = {EvoSLD: Automated Neural Scaling Law Discovery With Large Language Models},
  author  = {Lin, Haowei et al},
  journal = {arXiv preprint arXiv:2507.21184},
  year    = {2025}
}
```

-----

## 🙏 Acknowledgments

This project is built on the excellent **[OpenEvolve](https://github.com/codelion/openevolve)** evolutionary coding framework.