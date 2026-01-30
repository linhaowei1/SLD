# Can Language Models Discover Scaling Laws?

[![arXiv](https://img.shields.io/badge/arXiv-2507.21184-b31b1b.svg)](https://arxiv.org/abs/2507.21184)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SLDBench-blue)](https://huggingface.co/datasets/pkuHaowei/sldbench)
[![Leaderboard](https://img.shields.io/badge/🏆-Leaderboard-gold)](https://linhaowei1.github.io/scaling_law_discovery)
[![Harbor](https://img.shields.io/badge/Harbor-SLDBench-orange)](https://github.com/laude-institute/harbor-datasets/tree/main/datasets/sldbench)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official repository for the paper:** *"Can Language Models Discover Scaling Laws?"*

## 📰 News

- **[2026.01.26]** 🎉 Our paper has been accepted at **ICLR 2026**!
- **[2026.01.20]** 📝 Check out our **[main blog post](https://linhaowei1.github.io/scaling_law_discovery/blog/)** for an accessible overview of our work!

---

**SLDAgent** is an evolution-based AI agent that autonomously discovers scaling laws for large language models. This work introduces **SLDBench**, the first comprehensive benchmark for scaling law discovery, and demonstrates that AI agents can uncover laws that are more accurate and conceptually sound than their human-derived counterparts.

The agent co-optimizes both the **symbolic formula** of a scaling law and the **parameter-fitting algorithm**, enabling it to explore complex relationships and achieve superhuman performance in predicting model behavior at scale.

---

## 🔗 Quick Links

| Resource | Link |
|:---------|:-----|
| 📄 **Paper** | [arXiv:2507.21184](https://arxiv.org/abs/2507.21184) |
| 📊 **Dataset** | [SLDBench on Hugging Face](https://huggingface.co/datasets/pkuHaowei/sldbench) |
| 🏆 **Leaderboard** | [linhaowei1.github.io/scaling_law_discovery](https://linhaowei1.github.io/scaling_law_discovery) |
| 🚢 **Harbor Adapter** | [harbor-datasets/sldbench](https://github.com/laude-institute/harbor-datasets/tree/main/datasets/sldbench) |
| 🔧 **OpenEvolve Framework** | [github.com/codelion/openevolve](https://github.com/codelion/openevolve) |

---

## 🔬 Overview

Scaling laws are fundamental to understanding and predicting the behavior of large language models as they scale in size, data, and compute. However, discovering these laws has traditionally been a manual, labor-intensive process requiring significant domain expertise.

**Key Contributions:**
- **SLDAgent**: An AI agent that autonomously discovers scaling laws through evolutionary search
- **SLDBench**: A comprehensive benchmark containing 8 diverse scaling law discovery tasks
- **Superhuman Performance**: Agent-discovered laws outperform human expert baselines on multiple tasks
- **Open-Ended Discovery**: Agents can discover novel scaling law formulations not present in existing literature

---

## 🚢 Running SLDBench on General Code Agents

SLDBench has been integrated as an **adapter** in [Terminal-Bench Harbor](https://github.com/laude-institute/harbor-datasets/tree/main/datasets/sldbench), enabling evaluation of general-purpose code agents on scaling law discovery tasks.

To run SLDBench on your own agent:

1. **Follow the Terminal-Bench documentation**: Visit [tbench.ai](https://www.tbench.ai/) to learn about the Harbor evaluation framework
2. **Use the SLDBench adapter**: The adapter is available at [harbor-datasets/sldbench](https://github.com/laude-institute/harbor-datasets/tree/main/datasets/sldbench)
3. **Submit to the leaderboard**: View results and rankings at our [Leaderboard](https://linhaowei1.github.io/scaling_law_discovery)

---

## 📦 SLDBench: The Benchmark

**SLDBench** is the first comprehensive benchmark for scaling law discovery, curated from over 5,000 LLM training experiments from existing research literature. The benchmark evaluates an agent's ability to:

1. **Analyze experimental data** from LLM training runs
2. **Hypothesize functional forms** (power laws, mixture models, etc.)
3. **Optimize parameters** to fit the observed data
4. **Extrapolate accurately** to unseen regimes (larger models, more data, etc.)

### Tasks

| Task | Description | Config File |
| :--- | :--- | :--- |
| **Parallel Scaling Law** | Models the effect of parallelism P and model size N on loss | `configs/parallel_scaling_law.yaml` |
| **Vocabulary Scaling Law** | Models unigram-normalized loss as a function of non-vocabulary model size N, vocabulary size V, and dataset size D | `configs/vocab_scaling_law.yaml` |
| **SFT Scaling Law** | Models supervised fine-tuning loss based on dataset size D across various base models | `configs/sft_scaling_law.yaml` |
| **Domain Mixture Scaling Law** | Models pre-training loss for domains based on their proportion in the training mixture | `configs/domain_mixture_scaling_law.yaml` |
| **MoE Scaling Law** | Models loss in relation to network size N and number of experts E in Mixture-of-Experts architectures | `configs/moe_scaling_law.yaml` |
| **Data Constrained Scaling Law** | Models pre-training loss as a function of model size N, dataset size D, and unique tokens U | `configs/data_constrained_scaling_law.yaml` |
| **Learning Rate & Batch Size Scaling Law** | Models pre-training loss based on learning rate η, batch size b, dataset size D, and network size N | `configs/lr_bsz_scaling_law.yaml` |
| **U-Shaped Scaling Law** | An adversarial extrapolation regime probing non-monotonic (U-shaped or double-descent) scaling behaviors | `configs/easy_question_scaling_law.yaml` |

**Dataset:** All experimental data is centrally hosted on Hugging Face Hub at [pkuHaowei/sldbench](https://huggingface.co/datasets/pkuHaowei/sldbench).

**Evaluation Metrics:**
- **R² (Coefficient of Determination)**: Primary metric measuring extrapolation accuracy (1.0 = perfect)
- **NMSE (Normalized Mean Squared Error)**: Secondary error metric
- **NMAE (Normalized Mean Absolute Error)**: Secondary error metric

---

## 📋 Requirements

- **Python 3.13+**
- **[`uv`](https://docs.astral.sh/uv/)** package manager (recommended) or `pip`
- An **OpenAI-compatible** LLM API key (set `OPENAI_API_KEY`)
- macOS/Linux/Windows

> **Note**: `uv run` guarantees commands execute inside a synchronized project environment. If you prefer plain `pip`, you can adapt the commands accordingly.

---

## 🛠️ Installation

### Option 1: Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/linhaowei1/SLD.git
cd SLD

# Install dependencies
uv sync

# Set your LLM API key
export OPENAI_API_KEY="your_key_here"

# Optional: Configure non-default API endpoint
# export OPENAI_BASE_URL="https://your.openai.compatible.endpoint/v1"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_key_here"
# $env:OPENAI_BASE_URL="https://your.openai.compatible.endpoint/v1"
```

### Option 2: Using `pip`

```bash
# Clone the repository
git clone https://github.com/linhaowei1/SLD.git
cd SLD

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -e .

# Set your API key
export OPENAI_API_KEY="your_key_here"
```

---

## 🚀 Quick Start

### Run a Single Task

```bash
# Example: Data-Constrained Scaling Law discovery
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run openevolve-run \
  --config configs/data_constrained_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/data_constrained_scaling_law/run_1
```

### Run All Tasks in Batch

```bash
# Execute all 8 tasks across multiple models
bash scripts/run.sh
```

This will:
- Run each task **5 times** per model with different random seeds
- Save outputs to `results/{task_name}/{model}/run_{1,2,3,4,5}/`
- Store intermediate **checkpoints** during evolution
- Evaluate and save the **best program** from each run

---

## 📂 Project Structure

```
SLD/
├── configs/                     # Task configuration files
│   ├── data_constrained_scaling_law.yaml
│   ├── domain_mixture_scaling_law.yaml
│   ├── easy_question_scaling_law.yaml
│   ├── lr_bsz_scaling_law.yaml
│   ├── moe_scaling_law.yaml
│   ├── parallel_scaling_law.yaml
│   ├── sft_scaling_law.yaml
│   └── vocab_scaling_law.yaml
├── data_loader.py               # Unified data loading from HuggingFace
├── evaluator.py                 # Evaluation system with R², NMSE, NMAE metrics
├── init_program.py              # Initial scaling law template for evolution
├── results/                     # Experiment outputs (auto-generated)
│   └── {task_name}/
│       └── {model}/
│           └── run_{1,2,3,4,5}/
│               ├── checkpoints/     # Evolution checkpoints
│               └── best/            # Best discovered program
├── scripts/
│   └── run.sh                   # Batch execution script
├── pyproject.toml               # Python dependencies
├── CONTRIBUTING.md              # Guide for contributing new tasks
└── README.md
```

---

## 🏃 Usage

### Single Task Execution

```bash
export EVAL_TASK_NAME="data_constrained_scaling_law"
uv run openevolve-run \
  --config configs/data_constrained_scaling_law.yaml \
  init_program.py evaluator.py \
  --output results/data_constrained_scaling_law/run_1
```

### Batch Execution

```bash
bash scripts/run.sh
```

### Evaluating a Discovered Program

```bash
EVAL_TASK_NAME="data_constrained_scaling_law" \
uv run python evaluator.py \
  results/data_constrained_scaling_law/gpt-5/run_1/best/best_program.py
```

---

## ⚙️ Configuration Guide

Customize task behavior by editing YAML config files in `configs/`. Here's the actual structure used by SLDBench:

### Configuration File Structure

```yaml
# Root-level settings
max_iterations: 50              # Number of evolution generations
checkpoint_interval: 1          # Save checkpoint every N iterations
log_level: "INFO"               # Logging verbosity
random_seed: 42                 # Random seed for reproducibility

# LLM configuration
llm:
  api_base: "https://api.openai.com/v1"  # API endpoint
  max_tokens: 16384                       # Max tokens per request
  timeout: 240                            # API timeout (seconds)
  retries: 10                             # Retry attempts
  retry_delay: 10                         # Delay between retries

# Prompt configuration
prompt:
  system_message: |
    You are an expert in scaling laws...
  num_top_programs: 3           # Top programs to consider
  num_diverse_programs: 2       # Diverse programs for exploration
  use_template_stochasticity: true

# Database configuration for evolution
database:
  population_size: 100          # Population size per generation
  archive_size: 50              # Archive size for elites
  num_islands: 5                # Number of islands for island model
  migration_interval: 25        # Generations between migrations
  migration_rate: 0.1           # Fraction of population migrating
  elite_selection_ratio: 0.1    # Top % considered elite
  exploration_ratio: 0.2        # Exploration vs exploitation balance
  exploitation_ratio: 0.7       # Exploitation ratio
  feature_dimensions: ["combined_score", "complexity", "diversity"]
  feature_bins: 10

# Evaluator configuration
evaluator:
  timeout: 600                  # Evaluation timeout (seconds)
  max_retries: 3                # Max evaluation retries
  cascade_evaluation: false     # Enable cascading evaluation
  cascade_thresholds: [0.3, 0.6]
  parallel_evaluations: 4       # Concurrent evaluations
  use_llm_feedback: false       # Use LLM for feedback

# Evolution settings
diff_based_evolution: false
max_code_length: 100000
```

### Key Configuration Parameters

| Parameter | Location | Description | Default |
|:----------|:---------|:------------|:--------|
| `max_iterations` | Root | Number of evolution generations | 50 |
| `random_seed` | Root | Random seed for reproducibility | 42 |
| `llm.api_base` | `llm` | API endpoint URL | OpenAI |
| `llm.timeout` | `llm` | API timeout (seconds) | 240 |
| `llm.retries` | `llm` | API retry attempts | 10 |
| `database.population_size` | `database` | Population size per generation | 100 |
| `database.exploration_ratio` | `database` | Exploration vs exploitation balance | 0.2 |
| `evaluator.parallel_evaluations` | `evaluator` | Concurrent evaluations | 4 |
| `evaluator.timeout` | `evaluator` | Evaluation timeout (seconds) | 600 |

### Example: High-Thoroughness Configuration

```yaml
max_iterations: 100             # More generations

database:
  population_size: 200          # Larger population
  exploration_ratio: 0.3        # More exploration

evaluator:
  parallel_evaluations: 8       # Faster execution
```

---

## 💾 Data Loading

All experimental data is automatically loaded from [pkuHaowei/sldbench](https://huggingface.co/datasets/pkuHaowei/sldbench) on Hugging Face Hub.

**Data Splits:**
- `train`: Training data for parameter fitting
- `test`: Held-out test data for extrapolation evaluation

**Access Pattern:**
```python
from data_loader import load_data

# Automatically loads from HuggingFace
train_data = load_data(task_name="parallel_scaling_law", train=True)
test_data = load_data(task_name="parallel_scaling_law", train=False)

# Data is organized by groups
for group_key, (X, y) in train_data.items():
    print(f"Group: {group_key}, X shape: {X.shape}, y shape: {y.shape}")
```

---

## ➕ Adding Custom Scaling Laws

We welcome contributions of new scaling law discovery tasks! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for a comprehensive guide on:

- Preparing and formatting your experimental data
- Creating configuration files
- Registering tasks in the codebase
- Submitting to the benchmark

---

## 🆘 Troubleshooting

### Common Issues

| Issue | Solution |
|:------|:---------|
| **Import Errors** | Run `uv sync` to update dependencies |
| **Task Not Found** | Verify `EVAL_TASK_NAME` matches a task in `SUPPORTED_TASKS` (see `evaluator.py:22-31`) |
| **Data Loading Failures** | Check internet connection and access to HuggingFace Hub |
| **API Timeouts** | Increase `llm.timeout` and `llm.retries` in config YAML |
| **Script Permission Denied** | Run `chmod +x scripts/run.sh` or use `bash scripts/run.sh` |
| **Low R² Scores** | Tasks like `lr_bsz_scaling_law` and `easy_question_scaling_law` are extremely challenging; negative R² is expected even for expert baselines |

### Debug Tips

1. **Enable verbose logging**: Check `results/{task_name}/{model}/run_*/execution.log` for detailed execution logs
2. **Check checkpoint outputs**: Inspect intermediate checkpoints in `results/{task_name}/{model}/run_*/checkpoints/`
3. **Validate data loading**: Test `data_loader.py` independently with your task name
4. **Verify API access**: Test your `OPENAI_API_KEY` with a simple API call

---

## ❓ FAQ

**Q: Do I have to use OpenAI's API?**
A: No. Any OpenAI-compatible endpoint works. Set `api_base` in your YAML config under `llm` or use the `OPENAI_BASE_URL` environment variable.

**Q: Can I use `pip` instead of `uv`?**
A: Yes. Create a virtual environment, activate it, and install dependencies from `pyproject.toml` using `pip install -e .`

**Q: Where are the experiment results stored?**
A: Results are in `results/{task_name}/{model}/{run_id}/`. Each run contains checkpoints, logs, and `best/best_program.py`.

**Q: What does negative R² mean?**
A: Negative R² indicates predictions worse than simply using the mean. This is expected for challenging tasks like `lr_bsz_scaling_law` and `easy_question_scaling_law`, even for human expert baselines.

**Q: How do I interpret the evaluation metrics?**
A:
- **R² = 1.0**: Perfect extrapolation
- **R² = 0.0**: Predictions as good as the mean
- **R² < 0.0**: Predictions worse than the mean
- **NMSE/NMAE**: Lower is better; normalized error metrics

**Q: Can I use different LLM models?**
A: Yes. Pass the model name via command line using `--primary-model` flag (e.g., `--primary-model gpt-4`). Any model accessible via OpenAI-compatible API works.

**Q: How long does discovery take?**
A: Depends on configuration. With default settings (50 iterations, population 100), expect 30-60 minutes per task. Increase `evaluator.parallel_evaluations` to speed up.

---

## 📄 Citation

If you use SLDAgent or SLDBench in your research, please cite:

```bibtex
@article{lin2025languagemodelsdiscoverscaling,
  title={Can Language Models Discover Scaling Laws?},
  author={Haowei Lin and Haotian Ye and Wenzheng Feng and Quzhe Huang and Yujun Li and Hubert Lim and Zhengrui Li and Xiangyu Wang and Jianzhu Ma and Yitao Liang and James Zou },
  journal={arXiv preprint arXiv:2507.21184},
  year={2025},
  eprint={2507.21184},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.21184}
}
```

---

## 👥 Contributing

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines on:

- Adding new scaling law discovery tasks
- Improving the codebase
- Reporting issues and bugs

For questions or discussions, please open an issue on GitHub. For collaboration inquiries, contact [linhaowei@pku.edu.cn](mailto:linhaowei@pku.edu.cn).

---

## 🙏 Acknowledgments

This project is built on [OpenEvolve](https://github.com/codelion/openevolve), an excellent framework for evolution-based optimization. We thank the OpenEvolve team for their foundational work and collaboration—read our joint blog post: [SLDAgent + OpenEvolve](https://algorithmicsuperintelligence.ai/blog/openevolve-sldagent/index.html).

The SLDBench dataset is curated from over 5,000 LLM training experiments from numerous research papers and institutions. We gratefully acknowledge all original authors whose work contributed to this benchmark.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Questions?** Open an issue or contact [linhaowei@pku.edu.cn](mailto:linhaowei@pku.edu.cn)
