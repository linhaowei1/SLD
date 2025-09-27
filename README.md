# SLDAgent: Can Language Models Discover Scaling Laws? 🧬

**SLDAgent** is an evolution-based AI agent that **autonomously discovers scaling laws** for large language models. This work introduces **SLDBench**, a comprehensive benchmark for this new scientific discovery task, and demonstrates that `SLDAgent` can uncover laws that are more accurate and conceptually sound than their human-derived counterparts.

The agent co-optimizes both the **symbolic formula** of a scaling law and the **parameter-fitting algorithm**, enabling it to explore complex relationships and achieve superhuman performance in predicting model behavior at scale.

-----

## 📦 The SLDBench Benchmark

This project includes **SLDBench**, the first comprehensive benchmark for scaling law discovery, curated from over 5,000 LLM training experiments from existing literature.

| Task key | Config file |
| :--- | :--- |
| `parallel_scaling_law` | `configs/parallel_scaling_law.yaml` |
| `vocab_scaling_law` | `configs/vocab_scaling_law.yaml` |
| `sft_scaling_law` | `configs/sft_scaling_law.yaml` |
| `domain_mixture_scaling_law` | `configs/domain_mixture_scaling_law.yaml` |
| `moe_scaling_law` | `configs/moe_scaling_law.yaml` |
| `data_constrained_scaling_law` | `configs/data_constrained_scaling_law.yaml` |
| `lr_bsz_scaling_law` | `configs/lr_bsz_scaling_law.yaml` |

> Data is centrally hosted on Hugging Face Hub at [pkuHaowei/sldbench](https://huggingface.co/datasets/pkuHaowei/sldbench).

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
cd SLD

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
SLD/
├─ configs/                  # ⚙️ YAML configs (one per scaling law)
│  ├─ data_constrained_scaling_law.yaml
│  ├─ domain_mix_scaling_law.yaml
│  ├─ lr_and_bsz_scaling_law.yaml
│  ├─ moe_scaling_law.yaml
│  ├─ parallel_scaling_law.yaml
│  ├─ sft_scaling_law.yaml
│  └─ vocab_size_scaling_law.yaml
├─ data_loader.py            # ↔️ Unified data loading interface (from Hugging Face)
├─ evaluator.py              # ✅ Unified evaluation system
├─ init_program.py           # 🌱 Initial scaling-law template
├─ results/                  # 🏆 Outputs & checkpoints (created automatically)
└─ scripts/
   └─ run.sh                  # 🏃 Batch execution helper
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

### 1) Create a Config

Create `configs/your_law_name.yaml` and customize the settings (see the full template in the original README). Key sections include `llm`, `prompt`, `database`, and `evaluator`.

### 2) Prepare and Host Data

Upload your data to Hugging Face Hub. The data should be structured with appropriate feature and target columns following the existing schema patterns.

### 3) Register the Task Schema

Add your task schema to the `TASK_SCHEMA_MAP` dictionary in `data_loader.py`:

```python
TASK_SCHEMA_MAP = {
    # ... existing tasks ...
    "your_law_name": {
        "feature_names": ["feature1", "feature2"],
        "target_name": "target_variable",
    },
}
```

### 4) Register the Task

Add your task to the `SUPPORTED_TASKS` set in `evaluator.py`:

```python
SUPPORTED_TASKS = {
    # ... existing tasks ...
    "your_law_name",
}
```

### 5) (Optional) Add to Batch Script

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

  * Data is centrally hosted on Hugging Face Hub at [pkuHaowei/sldbench](https://huggingface.co/datasets/pkuHaowei/sldbench)
  * The unified `data_loader.py` automatically loads data based on the task name and predefined schema


## 🆘 Troubleshooting

  * **Import Errors**: Run `uv sync` to ensure your environment is up-to-date.
  * **Task Not Found**: Check that `EVAL_TASK_NAME` matches a task key in `SUPPORTED_TASKS` in `evaluator.py`.
  * **Data Loading Issues**: Verify internet connection and access to Hugging Face Hub repository `pkuHaowei/sldbench`.
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

If you use SLDAgent or SLDBench in your academic work, please cite the paper:

```bibtex
@article{lin2026SLD,
  title   = {Can Language Models Discover Scaling Laws?},
  author  = {Lin, Haowei et al.},
  year    = {2025}
}
```

-----

## 🙏 Acknowledgments

This project is built on the excellent **[OpenEvolve](https://github.com/codelion/openevolve)**.