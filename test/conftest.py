"""Pytest configuration and shared fixtures for moe-router-study tests."""

from collections.abc import Generator
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch as th
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def small_tensor() -> th.Tensor:
    """Create a small tensor for testing."""
    return th.randn(2, 3, 4)


@pytest.fixture
def router_logits() -> th.Tensor:
    """Create sample router logits tensor for MoE testing."""
    # Shape: (batch_size=2, seq_len=3, num_experts=8)
    return th.randn(2, 3, 8)


@pytest.fixture
def mock_hf_offline():
    """Mock HuggingFace to work in offline mode."""
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
        yield


@pytest.fixture
def tiny_tokenizer():
    """Create a minimal tokenizer for testing."""
    # Use a very small, fast tokenizer that should be cached
    try:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        return tokenizer
    except Exception:
        # Fallback to a mock-like tokenizer if the test tokenizer isn't available
        class MockTokenizer:
            def apply_chat_template(self, conversation, tokenize=False):
                if tokenize:
                    return [1, 2, 3]  # Mock token IDs
                return "mock chat template output"

            @property
            def pad_token(self):
                return "<pad>"

            @property
            def eos_token(self):
                return "</s>"

        return MockTokenizer()


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


@pytest.fixture
def mock_dataset_path(temp_dir: Path):
    """Create a mock dataset directory structure."""
    dataset_dir = temp_dir / "datasets" / "lmsys" / "lmsys-chat-1m"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


class TestConfig:
    """Test configuration constants."""

    # Small tensor sizes for testing
    BATCH_SIZE = 2
    SEQ_LEN = 4
    HIDDEN_DIM = 8
    NUM_EXPERTS = 4

    # Test model configurations
    TEST_MODEL_CONFIGS = {
        "test_model": {
            "hf_name": "test/model",
            "total_steps": 1000,
            "total_tokens": 5000,
        }
    }


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return TestConfig()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "requires_network: marks tests that require network access"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and content."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark network-dependent tests
        if any(
            keyword in item.name.lower()
            for keyword in ["network", "download", "fetch", "hub"]
        ):
            item.add_marker(pytest.mark.requires_network)

        # Mark GPU-dependent tests
        if any(keyword in item.name.lower() for keyword in ["gpu", "cuda"]):
            item.add_marker(pytest.mark.requires_gpu)
