"""Pytest configuration and shared fixtures (lean, no heavy mocking)."""

from collections.abc import Generator
from pathlib import Path
import tempfile

import pytest
import torch as th


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_device() -> str:
    """Return a device string for testing (CPU only)."""
    return "cpu"


# Add minimal ModelConfig fixture used by tests
@pytest.fixture
def mock_model_config():
    from core.model import ModelConfig

    return ModelConfig(hf_name="test/model", revision_format="step{}")


@pytest.fixture
def sample_tensor_3d() -> th.Tensor:
    """Create a sample 3D tensor (batch=2, layers=3, experts=4)."""
    return th.rand(2, 3, 4)


@pytest.fixture
def sample_bool_tensor_3d() -> th.Tensor:
    """Create a sample 3D boolean tensor."""
    tensor = th.zeros(2, 3, 4, dtype=th.bool)
    # Set some positions to True
    tensor[0, 0, 0] = True
    tensor[0, 1, 2] = True
    tensor[1, 2, 1] = True
    return tensor


@pytest.fixture
def sample_circuits_tensor() -> th.Tensor:
    """Create a sample circuits tensor (circuits=2, layers=3, experts=4)."""
    circuits = th.zeros(2, 3, 4, dtype=th.bool)
    # Circuit 0: experts 0,1 in layer 0; expert 2 in layer 1
    circuits[0, 0, 0] = True
    circuits[0, 0, 1] = True
    circuits[0, 1, 2] = True
    # Circuit 1: expert 3 in layer 0; experts 1,3 in layer 2
    circuits[1, 0, 3] = True
    circuits[1, 2, 1] = True
    circuits[1, 2, 3] = True
    return circuits


@pytest.fixture
def sample_router_logits() -> th.Tensor:
    """Create sample router logits (batch=2, layers=3, experts=4)."""
    return th.randn(2, 3, 4)


@pytest.fixture
def sample_activation_file_data() -> dict:
    """Sample activation file content for tests."""
    return {"topk": 2, "router_logits": th.randn(10, 3, 8)}


@pytest.fixture
def sample_weight_file_data() -> dict:
    """Sample weight file content for tests."""
    return {
        "checkpoint_idx": 0,
        "num_tokens": 1_000_000,
        "step": 1000,
        "topk": 2,
        "weights": {0: th.randn(8, 16), 1: th.randn(8, 16)},
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and paths (CPU-only)."""
    # Ensure we're using CPU for all tests
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    # Write outputs to test folders
    monkeypatch.setattr("exp.BASE_OUTPUT_DIR", "test_output", raising=False)
    monkeypatch.setattr("viz.BASE_FIGURE_DIR", "test_figures", raising=False)
    
    # Set a default test experiment name
    monkeypatch.setattr("exp.get_experiment_dir", lambda name=None: "test_output", raising=False)
    monkeypatch.setattr("viz.get_figure_dir", lambda experiment_name=None: "test_figures", raising=False)


@pytest.fixture
def mock_wandb():
    """Mock wandb/trackio components (allowed)."""
    from unittest.mock import MagicMock, patch

    with patch("trackio.Run") as mock_run:
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        yield mock_run_instance
