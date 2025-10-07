"""Test utilities for moe-router-study tests."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import torch as th


def create_mock_hf_refs(branches: List[str]) -> MagicMock:
    """Create a mock HuggingFace refs object with the given branch names."""
    mock_refs = MagicMock()
    mock_branches = []
    
    for branch_name in branches:
        mock_branch = MagicMock()
        mock_branch.name = branch_name
        mock_branches.append(mock_branch)
    
    mock_refs.branches = mock_branches
    return mock_refs


def create_synthetic_router_logits(
    batch_size: int = 2, 
    seq_len: int = 4, 
    num_experts: int = 8,
    seed: int = 42
) -> th.Tensor:
    """Create synthetic router logits for testing."""
    th.manual_seed(seed)
    return th.randn(batch_size, seq_len, num_experts)


def create_synthetic_activations(
    batch_size: int = 2,
    seq_len: int = 4, 
    hidden_dim: int = 64,
    seed: int = 42
) -> th.Tensor:
    """Create synthetic activation tensors for testing."""
    th.manual_seed(seed)
    return th.randn(batch_size, seq_len, hidden_dim)


def assert_tensor_shape(tensor: th.Tensor, expected_shape: tuple):
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_dtype(tensor: th.Tensor, expected_dtype: th.dtype):
    """Assert that a tensor has the expected dtype."""
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"


def create_mock_conversation_data(num_conversations: int = 5) -> List[Dict[str, Any]]:
    """Create mock conversation data for testing."""
    conversations = []
    for i in range(num_conversations):
        conversation = [
            {"role": "user", "content": f"User message {i}"},
            {"role": "assistant", "content": f"Assistant response {i}"},
        ]
        conversations.append({"conversation": conversation})
    return conversations


def create_mock_dataset_sample(text_samples: List[str]) -> List[Dict[str, str]]:
    """Create mock dataset samples with text field."""
    return [{"text": text} for text in text_samples]


class MockHuggingFaceModel:
    """Mock HuggingFace model for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self
    
    def __call__(self, *args, **kwargs):
        # Return mock outputs
        batch_size = args[0].shape[0] if args else 1
        seq_len = args[0].shape[1] if args else 10
        vocab_size = self.config.get("vocab_size", 1000)
        
        return MagicMock(
            logits=th.randn(batch_size, seq_len, vocab_size),
            hidden_states=th.randn(batch_size, seq_len, 768),
        )


def skip_if_no_gpu():
    """Decorator to skip tests if no GPU is available."""
    import pytest
    return pytest.mark.skipif(
        not th.cuda.is_available(),
        reason="GPU not available"
    )


def skip_if_no_network():
    """Decorator to skip tests if no network is available."""
    import pytest
    import socket
    
    def check_network():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    return pytest.mark.skipif(
        not check_network(),
        reason="Network not available"
    )


def parametrize_dtypes():
    """Parametrize tests across common PyTorch dtypes."""
    import pytest
    return pytest.mark.parametrize(
        "dtype", 
        [th.float32, th.float16, th.bfloat16]
    )


def parametrize_devices():
    """Parametrize tests across available devices."""
    import pytest
    devices = ["cpu"]
    if th.cuda.is_available():
        devices.append("cuda")
    
    return pytest.mark.parametrize("device", devices)
