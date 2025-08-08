"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch as th


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_device() -> str:
    """Return a mock device string for testing."""
    return "cpu"


@pytest.fixture
def sample_tensor_3d() -> th.Tensor:
    """Create a sample 3D tensor for testing (batch_size=2, num_layers=3, num_experts=4)."""
    return th.rand(2, 3, 4)


@pytest.fixture
def sample_bool_tensor_3d() -> th.Tensor:
    """Create a sample 3D boolean tensor for testing."""
    tensor = th.zeros(2, 3, 4, dtype=th.bool)
    # Set some random positions to True
    tensor[0, 0, 0] = True
    tensor[0, 1, 2] = True
    tensor[1, 2, 1] = True
    return tensor


@pytest.fixture
def sample_circuits_tensor() -> th.Tensor:
    """Create a sample circuits tensor for testing (num_circuits=2, num_layers=3, num_experts=4)."""
    circuits = th.zeros(2, 3, 4, dtype=th.bool)
    # Circuit 0: experts 0,1 in layer 0, expert 2 in layer 1
    circuits[0, 0, 0] = True
    circuits[0, 0, 1] = True
    circuits[0, 1, 2] = True
    # Circuit 1: expert 3 in layer 0, experts 1,3 in layer 2
    circuits[1, 0, 3] = True
    circuits[1, 2, 1] = True
    circuits[1, 2, 3] = True
    return circuits


@pytest.fixture
def sample_router_logits() -> th.Tensor:
    """Create sample router logits for testing."""
    # Shape: (batch_size=2, num_layers=3, num_experts=4)
    return th.randn(2, 3, 4)


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    from core.model import ModelConfig
    
    config = ModelConfig(
        hf_name="test/model",
        tokenizer_has_padding_token=True
    )
    return config


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint."""
    from core.model import Checkpoint, ModelConfig
    
    config = ModelConfig(hf_name="test/model")
    return Checkpoint(step=1000, num_tokens=1000000, model_config=config)


@pytest.fixture
def mock_transformers():
    """Mock transformers library components."""
    with patch('transformers.AutoModelForCausalLM') as mock_model, \
         patch('transformers.AutoTokenizer') as mock_tokenizer, \
         patch('transformers.utils.logging.disable_progress_bar'):
        
        # Configure mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': th.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': th.tensor([[1, 1, 1], [1, 1, 0]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Configure mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'model_instance': mock_model_instance,
            'tokenizer_instance': mock_tokenizer_instance
        }


@pytest.fixture
def mock_datasets():
    """Mock datasets library components."""
    with patch('datasets.load_dataset') as mock_load:
        # Create a mock dataset that yields text samples
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([
            "Sample text 1",
            "Sample text 2", 
            "Sample text 3"
        ])
        mock_load.return_value = {"text": mock_dataset}
        yield mock_load


@pytest.fixture
def mock_nnterp():
    """Mock nnterp library components."""
    with patch('nnterp.StandardizedTransformer') as mock_transformer:
        mock_instance = MagicMock()
        mock_instance.layers_with_routers = [0, 2, 4]
        mock_instance.router_probabilities.get_top_k.return_value = 2
        mock_instance.topk = 2
        mock_instance.routers = {
            0: MagicMock(weight=th.randn(8, 512)),
            2: MagicMock(weight=th.randn(8, 512)),
            4: MagicMock(weight=th.randn(8, 512))
        }
        mock_instance.layers = [MagicMock() for _ in range(6)]
        mock_instance.mlps = [MagicMock() for _ in range(6)]
        mock_instance.self_attn = [MagicMock() for _ in range(6)]
        
        # Configure MLP experts
        for i, mlp in enumerate(mock_instance.mlps):
            if i in [0, 2, 4]:  # MoE layers
                mlp.experts = [MagicMock() for _ in range(8)]
                for expert in mlp.experts:
                    expert.down_proj.weight = th.randn(512, 1024)
            else:  # Dense layers
                mlp.down_proj.weight = th.randn(512, 1024)
        
        # Configure attention layers
        for attn in mock_instance.self_attn:
            attn.out_proj.weight = th.randn(512, 512)
        
        mock_transformer.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_wandb():
    """Mock wandb/trackio components."""
    with patch('trackio.Run') as mock_run:
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        yield mock_run_instance


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib components."""
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.close') as mock_close, \
         patch('matplotlib.pyplot.scatter') as mock_scatter, \
         patch('matplotlib.pyplot.plot') as mock_plot, \
         patch('matplotlib.pyplot.bar') as mock_bar:
        
        yield {
            'savefig': mock_savefig,
            'close': mock_close,
            'scatter': mock_scatter,
            'plot': mock_plot,
            'bar': mock_bar
        }


@pytest.fixture
def sample_activation_file_data() -> dict:
    """Create sample activation file data for testing."""
    return {
        'topk': 2,
        'router_logits': th.randn(10, 3, 8)  # 10 tokens, 3 layers, 8 experts
    }


@pytest.fixture
def sample_weight_file_data() -> dict:
    """Create sample weight file data for testing."""
    return {
        'checkpoint_idx': 0,
        'num_tokens': 1000000,
        'step': 1000,
        'topk': 2,
        'weights': {
            0: th.randn(8, 512),
            2: th.randn(8, 512),
            4: th.randn(8, 512)
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and paths."""
    # Ensure we're using CPU for all tests
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    
    # Set test-specific paths
    monkeypatch.setattr("exp.OUTPUT_DIR", "test_output")
    monkeypatch.setattr("viz.FIGURE_DIR", "test_figures")

