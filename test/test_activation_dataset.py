"""Tests for the activation dataset implementation."""

from unittest.mock import patch

import pytest
import torch as th

from exp.activation_dataset import (
    ActivationDataset,
    NoDataFilesError,
    create_activation_dataloader,
    get_expert_indices_from_logits,
)


@pytest.fixture
def mock_file_system():
    """Mock file system with activation files."""
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=["0.pt", "1.pt", "2.pt"]),
        patch("torch.load") as mock_load,
    ):
        # Create mock data for each file
        def load_side_effect(path):
            batch_size = 2
            seq_len = 3
            num_experts = 4
            top_k = 2

            # Create router logits
            router_logits = th.randn(batch_size, seq_len, num_experts)

            # Create tokens
            tokens = [["token1", "token2", "token3"] for _ in range(batch_size)]

            return {
                "topk": top_k,
                "router_logits": router_logits,
                "tokens": tokens,
            }

        mock_load.side_effect = load_side_effect
        yield


def test_activation_dataset_init(mock_file_system):
    """Test ActivationDataset initialization."""
    dataset = ActivationDataset()
    assert len(dataset) == 3
    assert dataset.file_indices == [0, 1, 2]
    assert dataset.top_k == 2


def test_activation_dataset_getitem(mock_file_system):
    """Test ActivationDataset __getitem__."""
    dataset = ActivationDataset()
    item = dataset[0]

    assert "topk" in item
    assert "router_logits" in item
    assert "tokens" in item
    assert "file_idx" in item

    assert item["topk"] == 2
    assert item["router_logits"].shape == (2, 3, 4)
    assert len(item["tokens"]) == 2
    assert item["file_idx"] == 0


def test_activation_dataset_transform(mock_file_system):
    """Test ActivationDataset with transform."""

    def transform(item):
        return item["router_logits"]

    dataset = ActivationDataset(transform=transform)
    item = dataset[0]

    assert isinstance(item, th.Tensor)
    assert item.shape == (2, 3, 4)


def test_activation_dataset_no_files():
    """Test ActivationDataset with no files."""
    with (
        patch("os.path.isdir", return_value=True),
        patch("os.listdir", return_value=[]),
        pytest.raises(NoDataFilesError),
    ):
        ActivationDataset()


def test_activation_dataset_missing_dir():
    """Test ActivationDataset with missing directory."""
    with patch("os.path.isdir", return_value=False), pytest.raises(FileNotFoundError):
        ActivationDataset()


def test_create_activation_dataloader(mock_file_system):
    """Test create_activation_dataloader."""
    dataloader, top_k = create_activation_dataloader(batch_size=2)

    assert top_k == 2
    assert dataloader.batch_size == 2
    assert len(dataloader.dataset) == 3


def test_get_expert_indices_from_logits():
    """Test get_expert_indices_from_logits."""
    # Create test data
    batch_size = 2
    seq_len = 3
    num_experts = 4
    top_k = 2

    # Create router logits with known values
    router_logits = th.zeros(batch_size, seq_len, num_experts)
    router_logits[0, 0, 0] = 10.0  # Expert 0 should be selected
    router_logits[0, 0, 2] = 8.0  # Expert 2 should be selected
    router_logits[0, 1, 1] = 9.0  # Expert 1 should be selected
    router_logits[0, 1, 3] = 7.0  # Expert 3 should be selected

    # Get expert activations and indices
    expert_activations, expert_indices = get_expert_indices_from_logits(
        router_logits, top_k
    )

    # Check shapes
    assert expert_activations.shape == (batch_size, seq_len, num_experts)
    assert expert_indices.shape == (batch_size, seq_len, top_k)

    # Check expert activations
    assert expert_activations[0, 0, 0].item()
    assert not expert_activations[0, 0, 1].item()
    assert expert_activations[0, 0, 2].item()
    assert not expert_activations[0, 0, 3].item()

    assert not expert_activations[0, 1, 0].item()
    assert expert_activations[0, 1, 1].item()
    assert not expert_activations[0, 1, 2].item()
    assert expert_activations[0, 1, 3].item()

    # Check expert indices
    assert expert_indices[0, 0, 0].item() == 0
    assert expert_indices[0, 0, 1].item() == 2

    assert expert_indices[0, 1, 0].item() == 1
    assert expert_indices[0, 1, 1].item() == 3
