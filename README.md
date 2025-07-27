# MoE Router Study 🔬

A minimal research project for studying routing patterns in Mixture of Experts (MoE) models.

## 🎯 Research Goals

This project provides basic primitives for investigating:

1. **Router Pattern Analysis**: How do routing decisions correlate across different layers in MoE models?
2. **Expert Specialization**: Do experts develop specialized functions for different types of data?
3. **Subspace Relationships**: How do router weight subspaces relate to expert weight subspaces?
4. **Training Evolution**: How do routing patterns change during model training?

## 🏗️ Project Structure

```
moe-router-study/
├── core/                   # Core modules
│   ├── model_wrapper.py    # MoE model interface using nnterp
│   └── activation_collector.py  # Activation collection utilities
└── pyproject.toml          # Minimal dependencies (uv compatible)
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

```python
from core import MoEModelWrapper, ActivationCollector

# Load model
model = MoEModelWrapper("allenai/OLMoE-1B-7B-0924")
collector = ActivationCollector(model)

# Collect layer activations
prompts = ["Your text here", "Another example"]
layer_acts = collector.collect_layer_activations(prompts)

# Compute router activations from layer activations
router_acts = collector.collect_router_activations(layer_acts)
```

## 📊 Key Features

### Model Support
- **Primary Focus**: OLMoE (AllenAI's open MoE model)
- **Framework**: Built on nnterp for clean PyTorch model tracing

### Core Functionality
- **MoEModelWrapper**: Interface for loading and working with MoE models
- **ActivationCollector**: Collect layer activations and compute router activations
- **Minimal Dependencies**: Only essential packages included

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
