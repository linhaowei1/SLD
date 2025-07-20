# SLD - Scaling Law Discovery

An AI-powered scaling law discovery framework that uses OpenEvolve evolutionary programming to automatically discover and optimize scaling law functions for LLM fine-tuning.

## 📋 Project Overview

This project combines two core components:
- **Scaling Law Discovery Modules**: Multiple scaling law implementations including data-constrained, rectified, vocabulary, domain mixture, and mixture-of-experts scaling laws
- **OpenEvolve**: A powerful evolutionary programming framework for automated algorithm optimization and discovery

Through OpenEvolve's evolutionary algorithms, the project can automatically discover more accurate scaling law functions to predict LLM performance across different training data scales and configurations.

## 🎯 Key Features

- **Automated Scaling Law Discovery**: Uses evolutionary algorithms to automatically optimize scaling law function forms and parameters
- **Multi-Dataset Evaluation**: Validates performance on real LLM fine-tuning datasets including FLAN, Gigaword, and WMT19
- **Robustness Optimization**: Ensures discovered scaling laws perform well across different model families (T5, GPT, etc.) and data scales
- **Real-time Monitoring**: Supports checkpoint saving, evolution process visualization, and performance tracking

## 🏗️ Project Structure

```
SLD/
├── README.md                        # Project documentation
├── data_loader.py                   # Common data loading utilities
├── rectified_scaling_law/           # Rectified scaling law implementation
│   ├── config.yaml                 # OpenEvolve configuration
│   ├── evaluator.py                # Scaling law performance evaluator
│   ├── init_program.py             # Initial scaling law implementation
│   └── data/                       # Real LLM fine-tuning datasets
│       ├── flan.csv               # FLAN dataset training curves
│       ├── gigaword.csv           # Gigaword dataset training curves
│       └── wikiword.csv           # WikiWord dataset training curves
├── data_constrained_scaling_law/   # Data-constrained scaling law
├── vocab_scaling_law/              # Vocabulary scaling law
├── domain_mixture_scaling_law/     # Domain mixture scaling law
├── moe_scaling_law/                # Mixture-of-experts scaling law
└── openevolve/                     # OpenEvolve evolutionary programming framework
    ├── openevolve/                 # Core framework code
    ├── examples/                   # Usage examples
    ├── configs/                    # Configuration templates
    └── ...
```

## 🚀 Quick Start

### Environment Setup

1. **Install OpenEvolve**:
```bash
cd openevolve
pip install -e .
```

2. **Configure API Key**:
```bash
export OPENAI_API_KEY=your-api-key-here
```
Supports all OpenAI-compatible API endpoints (OpenAI, Anthropic, local models, etc.)

### Running Scaling Law Discovery

1. **Navigate to a scaling law directory**:
```bash
cd rectified_scaling_law
# Or try other variants:
# cd data_constrained_scaling_law
# cd vocab_scaling_law
# cd domain_mixture_scaling_law
# cd moe_scaling_law
```

2. **Start the evolutionary process**:
```bash
python ../openevolve/openevolve-run.py init_program.py evaluator.py --config config.yaml --iterations 100
```

3. **Monitor progress**:
```bash
# View generated results
ls openevolve_output/

# View best program
cat openevolve_output/best/best_program.py
```
## 📊 Dataset Information

The project uses multiple real LLM fine-tuning datasets:

- **FLAN**: Instruction fine-tuning dataset containing various NLP tasks
- **Gigaword**: Text summarization dataset
- **WikiWord**: Wikipedia-based language modeling dataset

Each dataset contains loss values across different model sizes and training data scales for validating scaling law accuracy.

Data format:
- Training data scale: 200 to 1,638,400 samples  
- Model parameters: From millions to billions of parameters
- Model families: T5, GPT, and other mainstream architectures

## ⚙️ Configuration Details

### Key Configuration Parameters

```yaml
# Evolution parameters  
max_iterations: 50              # Maximum evolution iterations
random_seed: 42                 # Random seed for reproducibility

# LLM configuration
llm:
  models:
    - name: "o4-mini"          # Language model to use
      weight: 1.0              # Model weight
  max_tokens: 16384            # Maximum token count
  
# Evolution algorithm configuration
database:
  population_size: 100         # Population size
  archive_size: 50            # Archive size
  num_islands: 3              # Number of islands (parallel evolution)
  
# Evaluation configuration  
evaluator:
  timeout: 30                 # Evaluation timeout
  parallel_evaluations: 4     # Number of parallel evaluations
```

## 🔬 Evaluation Metrics

The project uses the following metrics to evaluate scaling law performance:

- **MSE (Mean Squared Error)**: Primary evaluation metric, lower is better
- **R² Score**: Fit quality assessment  
- **Pearson Correlation**: Correlation between predictions and true values
- **Cross-dataset Generalization**: Performance consistency across different datasets

## 📈 Initial Scaling Laws

The project starts with basic power law functions:

```python
def scaling_law_func(data_points, params):
    """
    Initial power law scaling function: loss = (a / (x + offset)^b + c)^d
    """
    x = np.asarray(data_points, dtype=float)
    loss = np.power(params[0] / np.power(x + 1e07, params[1]) + params[2], params[3])
    return loss
```

OpenEvolve automatically optimizes the mathematical form and parameter fitting algorithms of these functions.

## 📁 Output Results

After completion, the `openevolve_output/` directory will contain:

- `best/best_program.py`: The discovered best scaling law function
- `checkpoints/`: Checkpoints from each iteration
- `logs/`: Detailed evolution logs
- Various metadata and performance tracking files

## 🔧 Advanced Usage

### Resume from Checkpoint

```bash
python ../openevolve/openevolve-run.py init_program.py evaluator.py \
  --config config.yaml \
  --checkpoint openevolve_output/checkpoints/checkpoint_50 \
  --iterations 50
```

### Custom Evaluators

You can modify `evaluator.py` to:
- Add new datasets
- Introduce additional evaluation metrics
- Adjust evaluation weights

### Configuration Optimization

Adjust configuration based on computational resources:
- Increase `population_size` for better results
- Adjust `parallel_evaluations` to match CPU cores
- Set appropriate `timeout` to avoid hanging

## 🤝 Contributing

We welcome contributions and ideas:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenEvolve](https://github.com/codelion/openevolve) - Powerful evolutionary programming framework
- Contributors of real LLM datasets
- Open source community support and contributions
