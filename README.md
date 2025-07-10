# SLD - AI缩放律发现项目

一个基于AI的缩放律发现项目，专注于大语言模型（LLM）微调场景下的缩放律研究和发现。

## 📋 项目简介

SLD（Scaling Law Discovery）是一个智能化的缩放律发现框架，旨在通过演化算法和机器学习技术自动发现和优化LLM训练过程中训练数据规模与模型损失之间的数学关系。

## 🚀 主要功能

### 核心模块：rectified_scaling_law

项目的主要功能集中在 `rectified_scaling_law/` 模块中，该模块提供：

- **智能缩放律发现**：基于真实LLM微调数据自动发现最优缩放律函数
- **多数据集支持**：支持FLAN、Gigaword、WMT19等主流数据集
- **多模型族评估**：覆盖GPT、T5、BART、OPT、Phi等主要模型家族
- **演化优化算法**：通过OpenEvolve框架进行函数演化和参数优化
- **全面性能评估**：提供MSE、R²、MAPE等多种评估指标

## 📁 项目结构

```
SLD/
├── rectified_scaling_law/          # 🆕 主要功能模块
│   ├── config.yaml                 # 实验配置文件
│   ├── evaluator.py               # 缩放律评估器
│   ├── init_program.py            # 初始缩放律程序
│   ├── data/                      # 训练数据集
│   │   ├── flan.csv              # FLAN数据集
│   │   ├── gigaword.csv          # Gigaword数据集
│   │   └── wmt19.csv             # WMT19数据集
│   └── openevolve_output/         # 演化输出结果
│       ├── checkpoints/           # 训练检查点
│       └── logs/                  # 日志文件
├── openevolve/                    # OpenEvolve框架
├── README.md                      # 项目文档
├── LICENSE                        # 许可证
└── .gitignore                     # Git忽略文件
```

## 🔧 安装与环境配置

### 环境要求

- Python 3.8+
- NumPy, Pandas, SciPy
- scikit-learn
- PyYAML

### 安装步骤

```bash
# 克隆项目
git clone <repository-url>
cd SLD

# 安装依赖
pip install numpy pandas scipy scikit-learn pyyaml

# 进入主要工作目录
cd rectified_scaling_law
```

## 📊 数据集说明

项目包含三个主要数据集，每个数据集包含多个LLM模型在不同训练数据规模下的损失值：

### 支持的数据规模
- 训练数据大小：200 到 1,638,400 个样本
- 包含14个不同的数据规模点

### 支持的模型族
- **GPT系列**：GPT-2 (small/medium/large/xl)
- **T5系列**：T5 (small/base), mT5 (base/large)
- **BART系列**：BART (base/large), BART-CNN, BART-XSUM
- **OPT系列**：OPT (350M/1.3B/2.7B/6.7B)
- **其他**：Phi, LaMini, Cerebras-GPT, Switch Transformer等

### 数据集特点
- **FLAN数据集**：指令微调数据
- **Gigaword数据集**：文本摘要数据
- **WMT19数据集**：机器翻译数据

## 🚀 快速开始

### 1. 基础使用

```bash
# 运行初始缩放律程序
python init_program.py

# 评估特定缩放律程序
python evaluator.py path/to/scaling_law_program.py
```

### 2. 配置实验

编辑 `config.yaml` 文件来自定义实验参数：

```yaml
# 实验基础配置
max_iterations: 50
checkpoint_interval: 1
random_seed: 42

# LLM配置
llm:
  models:
    - name: "o4-mini"
      weight: 1.0
  max_tokens: 16384

# 评估器配置
evaluator:
  timeout: 30
  parallel_evaluations: 4
```

### 3. 自定义缩放律函数

实现您自己的缩放律函数：

```python
def scaling_law_func(data_points, params):
    """
    自定义缩放律函数
    
    Args:
        data_points: 训练数据规模数组
        params: 缩放律参数数组
        
    Returns:
        预测的损失值
    """
    # 实现您的缩放律逻辑
    pass

def fit_scaling_law(data_points, loss_values, initial_params=None):
    """
    缩放律参数拟合函数
    """
    # 实现参数优化逻辑
    pass
```

## 📈 评估指标

项目提供多种评估指标来衡量缩放律的性能：

- **MSE (均方误差)**：主要评估指标
- **RMSE (均方根误差)**：标准化误差测量
- **R² (决定系数)**：拟合优度
- **相关系数**：预测与真实值的线性相关性
- **MAPE (平均绝对百分比误差)**：相对误差测量

## 🔬 实验特性

### 演化算法支持
- 支持基于OpenEvolve的函数演化
- 多种演化策略：精英选择、探索开发平衡
- 并行评估加速训练过程

### 鲁棒性保证
- 数值稳定性检查
- 超时保护机制
- 异常处理和恢复

### 可扩展性
- 支持新数据集的添加
- 模块化设计便于功能扩展
- 配置驱动的实验管理

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用开源许可证，详见 `LICENSE` 文件。

## 📞 联系方式

如有问题或建议，请通过Issue联系我们。
