"""Tests for activation loading utilities."""
import os
from unittest.mock import patch

import pytest
import torch as th

from exp.activations import (
    load_activations,
    load_activations_and_indices_and_topk,
    load_activations_and_topk,
    load_activations_indices_tokens_and_topk,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class TestActivationLoading:
    """Tests for activation loading functions."""

    def test_basic_loading(self, temp_dir):
        """Test basic loading of activations."""
        # Create test data
        batch_size = 3
        num_layers = 2
        num_experts = 4
        topk = 2

        # Create a single router logits file
        router_logits = th.randn(batch_size, num_layers, num_experts)
        th.save({"router_logits": router_logits, "topk": topk}, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
        ):
            # Test load_activations_indices_tokens_and_topk
            activated_experts, indices, tokens, loaded_topk = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment"
            )
            assert activated_experts.shape == (batch_size, num_layers, num_experts)
            assert indices.shape == (batch_size, num_layers, topk)
            assert tokens is None
            assert loaded_topk == topk

            # Test load_activations_and_indices_and_topk
            activated_experts2, indices2, loaded_topk2 = load_activations_and_indices_and_topk(
                experiment_name="test_experiment"
            )
            assert th.all(activated_experts2 == activated_experts)
            assert th.all(indices2 == indices)
            assert loaded_topk2 == loaded_topk

            # Test load_activations_and_topk
            activated_experts3, loaded_topk3 = load_activations_and_topk(
                experiment_name="test_experiment"
            )
            assert th.all(activated_experts3 == activated_experts)
            assert loaded_topk3 == loaded_topk

            # Test load_activations
            activated_experts4 = load_activations(experiment_name="test_experiment")
            assert th.all(activated_experts4 == activated_experts)

    def test_multiple_files(self, temp_dir):
        """Test loading activations from multiple files."""
        # Create test data
        batch_size = 3
        num_layers = 2
        num_experts = 4
        topk = 2

        # Create multiple router logits files
        for i in range(3):
            router_logits = th.randn(batch_size, num_layers, num_experts)
            th.save({"router_logits": router_logits, "topk": topk}, temp_dir / f"{i}.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
        ):
            activated_experts, indices, tokens, loaded_topk = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment"
            )
            assert activated_experts.shape == (batch_size * 3, num_layers, num_experts)
            assert indices.shape == (batch_size * 3, num_layers, topk)
            assert tokens is None
            assert loaded_topk == topk

    def test_with_tokens(self, temp_dir):
        """Test loading activations with tokens."""
        # Create test data
        batch_size = 3
        num_layers = 2
        num_experts = 4
        topk = 2
        tokens = ["token1", "token2", "token3"]

        # Create a single router logits file with tokens
        router_logits = th.randn(batch_size, num_layers, num_experts)
        th.save(
            {"router_logits": router_logits, "topk": topk, "tokens": tokens},
            temp_dir / "0.pt",
        )

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
        ):
            activated_experts, indices, loaded_tokens, loaded_topk = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment"
            )
            assert activated_experts.shape == (batch_size, num_layers, num_experts)
            assert indices.shape == (batch_size, num_layers, topk)
            assert loaded_tokens == tokens
            assert loaded_topk == topk

    def test_device_placement(self, temp_dir):
        """Test device placement of loaded tensors."""
        # Create test data
        batch_size = 3
        num_layers = 2
        num_experts = 4
        topk = 2

        # Create a single router logits file
        router_logits = th.randn(batch_size, num_layers, num_experts)
        th.save({"router_logits": router_logits, "topk": topk}, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
        ):
            # Test with default device (cpu)
            activated_experts, indices, _, _ = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment"
            )
            assert activated_experts.device.type == "cpu"
            assert indices.device.type == "cpu"

            # Test with specified device (still cpu, but explicit)
            activated_experts, indices, _, _ = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment", device="cpu"
            )
            assert activated_experts.device.type == "cpu"
            assert indices.device.type == "cpu"


class TestActivationLoadingErrorHandling:
    """Tests for error handling in activation loading functions."""

    def test_directory_not_found(self):
        """Test handling of directory not found error."""
        with (
            patch("exp.activations.get_experiment_dir", return_value="/non/existent/path"),
            patch("exp.activations.get_router_logits_dir", return_value="/non/existent/path"),
            pytest.raises(FileNotFoundError, match="Activation directory not found"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_empty_directory(self, temp_dir):
        """Test handling of empty directory."""
        # Create an empty directory
        os.makedirs(temp_dir, exist_ok=True)

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(ValueError, match="No data files found"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_missing_topk_key(self, temp_dir):
        """Test handling of missing topk key."""
        # Create data without topk key
        data = {"router_logits": th.randn(3, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(KeyError, match="Missing 'topk' key"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_missing_router_logits_key(self, temp_dir):
        """Test handling of missing router_logits key."""
        # Create data without router_logits key
        data = {"topk": 2}
        th.save(data, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(KeyError, match="Missing 'router_logits' key"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_invalid_tensor_shapes(self, temp_dir):
        """Test handling of invalid tensor shapes."""
        # Create data with wrong number of dimensions
        data = {
            "topk": 2,
            "router_logits": th.randn(3, 4),  # Only 2D instead of 3D
        }
        th.save(data, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(ValueError, match="Expected router_logits to be 3D"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_zero_topk_value(self, temp_dir):
        """Test handling of zero topk value."""
        data = {"topk": 0, "router_logits": th.randn(3, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(ValueError, match="Invalid topk value"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_topk_larger_than_experts(self, temp_dir):
        """Test handling of topk larger than number of experts."""
        data = {
            "topk": 10,  # Larger than number of experts (4)
            "router_logits": th.randn(3, 2, 4),
        }
        th.save(data, temp_dir / "0.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(ValueError, match="topk .* cannot be greater than number of experts"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_file_not_found(self, temp_dir):
        """Test handling of file not found error."""
        # Use a non-existent directory
        non_existent_dir = os.path.join(temp_dir, "non_existent_dir")
        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=non_existent_dir),
            pytest.raises(FileNotFoundError, match="Activation directory not found"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_inconsistent_topk(self, temp_dir):
        """Test handling of inconsistent topk values across files."""
        # Create files with different topk values
        data1 = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        data2 = {"topk": 3, "router_logits": th.randn(3, 2, 4)}
        th.save(data1, temp_dir / "0.pt")
        th.save(data2, temp_dir / "1.pt")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(ValueError, match="Inconsistent topk values"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

    def test_non_contiguous_files(self, temp_dir):
        """Test handling of non-contiguous file indices."""
        # Create files with non-contiguous indices
        data1 = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        data2 = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        th.save(data1, temp_dir / "0.pt")
        th.save(data2, temp_dir / "2.pt")  # Skip 1.pt

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
        ):
            # Should only load the first file
            activated_experts, _, _, _ = load_activations_indices_tokens_and_topk(
                experiment_name="test_experiment"
            )
            assert activated_experts.shape[0] == 3  # Only one batch loaded

    def test_corrupted_file(self, temp_dir):
        """Test handling of corrupted file."""
        # Create a corrupted file
        with open(temp_dir / "0.pt", "w") as f:
            f.write("This is not a valid PyTorch file")

        with (
            patch("exp.activations.get_experiment_dir", return_value=str(temp_dir)),
            patch("exp.activations.get_router_logits_dir", return_value=str(temp_dir)),
            pytest.raises(RuntimeError, match="Failed to load router logits file"),
        ):
            load_activations_and_indices_and_topk(experiment_name="test_experiment")

