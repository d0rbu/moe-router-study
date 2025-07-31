# MoE Router Study

A research project studying Mixture of Experts (MoE) router mechanisms using the nnterp library for mechanistic interpretability.

## Project Structure

```
moe-router-study/
├── core/           # Core utilities and models
├── exp/            # Experiments and analysis scripts
├── viz/            # Visualization utilities
├── data/           # Data files (gitignored)
├── output/         # Output files and results (gitignored)
├── test/           # Test files
└── README.md
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/d0rbu/moe-router-study.git
cd moe-router-study
```

2. Install dependencies with uv:
```bash
# Install all dependencies including dev tools
uv sync --all-extras --dev

# Or install just the core dependencies
uv sync
```

3. Install pre-commit hooks (optional but recommended):
```bash
uv run pre-commit install
```

### Using nnterp

This project uses a custom fork of nnterp with additional features. The dependency is automatically installed from:
https://github.com/d0rbu/nnterp

Key features of nnterp used in this project:
- Unified interface for all transformer models
- Standardized naming conventions across architectures
- Built-in mechanistic interpretability tools
- Efficient activation collection and analysis

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=core --cov=exp --cov=viz

# Run only fast tests (skip slow integration tests)
uv run pytest -m "not slow"
```

### Code Quality

```bash
# Run linting
uv run ruff check .

# Run formatting
uv run ruff format .

# Run type checking
uv run mypy core/ exp/ viz/
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Linting & Type Checking**: Runs ruff and mypy on Python 3.9-3.12
- **Testing**: Runs pytest with coverage reporting
- **Security Scanning**: Runs safety and bandit security checks

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
