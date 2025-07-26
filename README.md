# MoE Router Study 🔬

A research project studying routing patterns in Mixture of Experts (MoE) models, with a focus on analyzing how expert activations correlate across layers and evolve throughout training.

## 🎯 Research Goals

This project investigates:

1. **Router Pattern Analysis**: How do routing decisions correlate across different layers in MoE models?
2. **Expert Specialization**: Do experts develop specialized functions for different types of data (math, code, general text)?
3. **Subspace Relationships**: How do router weight subspaces relate to expert weight subspaces?
4. **Training Evolution**: How do routing patterns change during model training?

## 🏗️ Project Structure

```
moe-router-study/
├── core/                   # Core analysis modules
│   ├── model_wrapper.py    # MoE model interface using nnterp
│   ├── activation_collector.py  # Activation collection utilities
│   ├── router_analyzer.py  # Router pattern analysis
│   └── expert_analyzer.py  # Expert weight analysis
├── data/                   # Data loading and preprocessing
│   ├── dataset_loader.py   # Load various datasets (FineWeb, math, code)
│   └── prompt_generator.py # Generate targeted prompts
├── exp/                    # Experiment scripts
│   └── basic_routing_analysis.py  # Basic routing analysis experiment
├── viz/                    # Visualization utilities
│   └── plotting.py         # Plotting functions for analysis
├── results/                # Experiment results and outputs
├── configs/                # Configuration files
└── pyproject.toml          # Project dependencies (uv compatible)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (tested on RTX 4090)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/d0rbu/moe-router-study.git
cd moe-router-study
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Quick Start

Run the basic routing analysis experiment:

```bash
python exp/basic_routing_analysis.py
```

This will:
- Load the OLMoE model using nnterp
- Collect routing activations on different datasets
- Analyze router weight patterns and correlations
- Generate visualizations
- Log results to Weights & Biases (optional)

## 📊 Key Features

### Model Support
- **Primary Focus**: OLMoE (AllenAI's open MoE model)
- **Framework**: Built on nnterp for clean PyTorch model tracing
- **Checkpoints**: Support for analyzing training history via model revisions

### Analysis Capabilities
- **Router Analysis**: Extract and analyze router weight matrices
- **Expert Analysis**: Study expert weight patterns and specialization
- **Correlation Analysis**: Measure cross-layer activation correlations
- **Subspace Analysis**: SVD/PCA analysis of weight subspaces
- **Routing Patterns**: Analyze expert usage and load balancing

### Data Support
- **Pretraining Data**: FineWeb samples
- **Math Data**: GSM8K, MATH, AQuA-RAT
- **Code Data**: GitHub code, HumanEval
- **Custom Prompts**: Capability-targeted prompt generation

## 🔬 Research Methods

### Router Pattern Analysis
```python
from core import MoEModelWrapper, RouterAnalyzer

# Load model
model = MoEModelWrapper("allenai/OLMoE-1B-7B-0924")
analyzer = RouterAnalyzer(model)

# Extract router weights
router_weights = analyzer.extract_all_router_weights()

# Compute similarity matrices
similarity = analyzer.compute_router_similarity_matrix(router_weights)

# Analyze subspaces
subspaces = analyzer.analyze_router_subspaces(router_weights)
```

### Expert Specialization Analysis
```python
from core import ExpertAnalyzer

expert_analyzer = ExpertAnalyzer(model)

# Extract expert weights
expert_weights = expert_analyzer.extract_all_expert_weights()

# Analyze router-expert alignment
alignment = expert_analyzer.analyze_expert_router_alignment(
    expert_weights, router_weights
)
```

### Activation Collection
```python
from core import ActivationCollector
from data import DatasetLoader

collector = ActivationCollector(model)
loader = DatasetLoader()

# Load different datasets
math_texts, _ = loader.load_math_qa_data(num_samples=100)
code_texts, _ = loader.load_code_data(num_samples=100)

# Collect router activations
math_activations = collector.collect_router_activations(math_texts)
code_activations = collector.collect_router_activations(code_texts)

# Analyze correlations
correlations = collector.compute_activation_correlations(math_activations)
```

## 📈 Visualization

The project includes comprehensive visualization tools:

- **Router Similarity Heatmaps**: Visualize similarity between router weights
- **Expert Usage Patterns**: Track expert activation across layers
- **Correlation Matrices**: Show cross-layer activation correlations
- **Interactive Plots**: Plotly-based interactive visualizations
- **Training Evolution**: Track changes across model checkpoints

## 🧪 Experiments

### Basic Routing Analysis
Analyzes fundamental routing patterns across different data types:
- Router weight similarity across layers
- Expert usage patterns for different datasets
- Cross-layer activation correlations
- Load balancing analysis

### Advanced Experiments (Planned)
- **Subspace Alignment Study**: Detailed analysis of router-expert subspace relationships
- **Training Dynamics**: Evolution of routing patterns during training
- **Intervention Studies**: Targeted expert activation experiments
- **Circuit Discovery**: Finding expert circuits for specific capabilities

## 📊 Integration with Weights & Biases

The project supports automatic logging to Weights & Biases:

```python
# Enable W&B logging
experiment = BasicRoutingAnalysis(
    model_name="allenai/OLMoE-1B-7B-0924",
    use_wandb=True,
    wandb_project="moe-routing-study"
)
```

Logged metrics include:
- Router similarity scores
- Expert usage statistics
- Routing entropy and load balance
- Correlation coefficients
- Visualizations and plots

## 🤝 Contributing

This is a research project. Contributions, suggestions, and discussions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- **OLMoE Paper**: [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
- **nnterp Library**: Built on [nnterp](https://pypi.org/project/nnterp/) for model interpretability
- **NNsight**: Underlying framework for PyTorch model tracing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AllenAI for releasing OLMoE and training checkpoints
- The nnterp and NNsight teams for interpretability tools
- The broader MoE research community

---

**Happy researching!** 🔬✨

