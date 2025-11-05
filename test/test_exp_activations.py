"""Tests for exp.activations module."""

from unittest.mock import patch

import pytest
import torch as th

from exp.activations import (
    load_activations,
    load_activations_and_indices_and_topk,
    load_activations_and_topk,
)
from test.test_utils import assert_tensor_shape_and_type


class TestLoadActivationsAndIndicesAndTopk:
    """Test load_activations_and_indices_and_topk function."""

    def test_basic_loading(self, temp_dir):
        """Test basic activation loading functionality."""
        # Create test data files
        test_data = []
        for i in range(3):
            data = {
                "topk": 2,
                "router_logits": th.randn(5, 3, 8),  # 5 tokens, 3 layers, 8 experts
            }
            file_path = temp_dir / f"{i}.pt"
            th.save(data, file_path)
            test_data.append(data)

        # Mock the ROUTER_LOGITS_DIR
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, activated_indices, top_k = (
                load_activations_and_indices_and_topk()
            )

        # Check outputs
        assert top_k == 2

        # Should concatenate all data files
        expected_batch_size = 5 * 3  # 5 tokens per file, 3 files
        assert_tensor_shape_and_type(
            activated_experts, (expected_batch_size, 3, 8), th.bool
        )
        assert_tensor_shape_and_type(
            activated_indices, (expected_batch_size, 3, 2)
        )  # topk=2

        # Check that activations are boolean
        assert activated_experts.dtype == th.bool

        # Check that indices are within valid range
        assert th.all(activated_indices >= 0)
        assert th.all(activated_indices < 8)  # num_experts

    def test_topk_consistency(self, temp_dir):
        """Test that topk values are consistent across files."""
        # Create files with same topk
        for i in range(2):
            data = {"topk": 3, "router_logits": th.randn(4, 2, 6)}
            th.save(data, temp_dir / f"{i}.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            _, _, top_k = load_activations_and_indices_and_topk()

        assert top_k == 3

    def test_empty_directory(self, temp_dir):
        """Test behavior with empty directory."""
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(ValueError, match="No data files found"):
                load_activations_and_indices_and_topk()

    def test_single_file(self, temp_dir):
        """Test loading from single file."""
        data = {"topk": 1, "router_logits": th.randn(10, 4, 12)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, activated_indices, top_k = (
                load_activations_and_indices_and_topk()
            )

        assert top_k == 1
        assert_tensor_shape_and_type(activated_experts, (10, 4, 12), th.bool)
        assert_tensor_shape_and_type(activated_indices, (10, 4, 1))

    def test_activation_logic(self, temp_dir):
        """Test that activation logic works correctly."""
        # Create router logits with known top-k pattern
        router_logits = th.tensor(
            [
                [[1.0, 3.0, 2.0, 0.0]],  # Top-2: indices 1, 2
                [[0.5, 0.1, 2.0, 1.5]],  # Top-2: indices 2, 3
            ]
        )

        data = {"topk": 2, "router_logits": router_logits}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, activated_indices, top_k = (
                load_activations_and_indices_and_topk()
            )

        # Check first sample
        assert activated_experts[0, 0, 1]  # Expert 1 should be active
        assert activated_experts[0, 0, 2]  # Expert 2 should be active
        assert not activated_experts[0, 0, 0]  # Expert 0 should not be active
        assert not activated_experts[0, 0, 3]  # Expert 3 should not be active

        # Check second sample
        assert activated_experts[1, 0, 2]  # Expert 2 should be active
        assert activated_experts[1, 0, 3]  # Expert 3 should be active
        assert not activated_experts[1, 0, 0]  # Expert 0 should not be active
        assert not activated_experts[1, 0, 1]  # Expert 1 should not be active

    def test_device_handling(self, temp_dir):
        """Test device handling in activation loading."""
        data = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            # Test with CPU device
            activated_experts, _, _ = load_activations_and_indices_and_topk(
                device="cpu"
            )
            assert activated_experts.device.type == "cpu"

    def test_file_numbering_gaps(self, temp_dir):
        """Test handling of gaps in file numbering."""
        # Create files 0.pt and 2.pt (missing 1.pt)
        for i in [0, 2]:
            data = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
            th.save(data, temp_dir / f"{i}.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, _, _ = load_activations_and_indices_and_topk()

        # Should only load file 0.pt (stops at first missing file)
        assert_tensor_shape_and_type(activated_experts, (3, 2, 4), th.bool)

    def test_large_batch_concatenation(self, temp_dir):
        """Test concatenation of multiple batches."""
        batch_sizes = [5, 3, 7, 2]
        total_batch_size = sum(batch_sizes)

        for i, batch_size in enumerate(batch_sizes):
            data = {"topk": 2, "router_logits": th.randn(batch_size, 3, 6)}
            th.save(data, temp_dir / f"{i}.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, activated_indices, _ = (
                load_activations_and_indices_and_topk()
            )

        assert_tensor_shape_and_type(
            activated_experts, (total_batch_size, 3, 6), th.bool
        )
        assert_tensor_shape_and_type(activated_indices, (total_batch_size, 3, 2))


class TestLoadActivationsAndTopk:
    """Test load_activations_and_topk function."""

    def test_wrapper_functionality(self, temp_dir):
        """Test that wrapper function works correctly."""
        data = {"topk": 3, "router_logits": th.randn(4, 2, 8)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, top_k = load_activations_and_topk()

        assert top_k == 3
        assert_tensor_shape_and_type(activated_experts, (4, 2, 8), th.bool)

    def test_consistency_with_full_function(self, temp_dir):
        """Test consistency with full loading function."""
        data = {"topk": 2, "router_logits": th.randn(5, 3, 6)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            # Call both functions
            full_experts, full_indices, full_topk = (
                load_activations_and_indices_and_topk()
            )
            wrapper_experts, wrapper_topk = load_activations_and_topk()

        # Should return same results
        assert th.equal(full_experts, wrapper_experts)
        assert full_topk == wrapper_topk

    def test_device_parameter(self, temp_dir):
        """Test device parameter in wrapper function."""
        data = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, _ = load_activations_and_topk(device="cpu")
            assert activated_experts.device.type == "cpu"


class TestLoadActivations:
    """Test load_activations function."""

    def test_simple_wrapper(self, temp_dir):
        """Test simple wrapper functionality."""
        data = {"topk": 2, "router_logits": th.randn(6, 4, 10)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts = load_activations()

        assert_tensor_shape_and_type(activated_experts, (6, 4, 10), th.bool)

    def test_consistency_with_full_function(self, temp_dir):
        """Test consistency with full loading function."""
        data = {"topk": 3, "router_logits": th.randn(4, 3, 8)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            full_experts, _, _ = load_activations_and_indices_and_topk()
            simple_experts = load_activations()

        assert th.equal(full_experts, simple_experts)

    def test_device_parameter_simple(self, temp_dir):
        """Test device parameter in simple wrapper."""
        data = {"topk": 1, "router_logits": th.randn(2, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts = load_activations(device="cpu")
            assert activated_experts.device.type == "cpu"


class TestActivationLoadingErrorHandling:
    """Test error handling in activation loading."""

    def test_corrupted_file_handling(self, temp_dir):
        """Test handling of corrupted data files."""
        # Create a corrupted file
        corrupted_file = temp_dir / "0.pt"
        corrupted_file.write_text("corrupted data")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(Exception):  # Should raise some kind of loading error
                load_activations_and_indices_and_topk()

    def test_missing_topk_key(self, temp_dir):
        """Test handling of missing topk key in data."""
        data = {
            "router_logits": th.randn(3, 2, 4)
            # Missing 'topk' key
        }
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(KeyError):
                load_activations_and_indices_and_topk()

    def test_missing_router_logits_key(self, temp_dir):
        """Test handling of missing router_logits key in data."""
        data = {
            "topk": 2
            # Missing 'router_logits' key
        }
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(KeyError):
                load_activations_and_indices_and_topk()

    def test_inconsistent_topk_values(self, temp_dir):
        """Test handling of inconsistent topk values across files."""
        # Create files with different topk values
        data1 = {"topk": 2, "router_logits": th.randn(3, 2, 4)}
        data2 = {"topk": 3, "router_logits": th.randn(3, 2, 4)}

        th.save(data1, temp_dir / "0.pt")
        th.save(data2, temp_dir / "1.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            # Should use the topk from the last file processed
            _, _, top_k = load_activations_and_indices_and_topk()
            assert top_k == 3  # From the last file

    def test_invalid_tensor_shapes(self, temp_dir):
        """Test handling of invalid tensor shapes."""
        # Create data with wrong number of dimensions
        data = {
            "topk": 2,
            "router_logits": th.randn(3, 4),  # Only 2D instead of 3D
        }
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(Exception):  # Should fail during topk operation
                load_activations_and_indices_and_topk()

    def test_zero_topk_value(self, temp_dir):
        """Test handling of zero topk value."""
        data = {"topk": 0, "router_logits": th.randn(3, 2, 4)}
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(Exception):  # Should fail during topk operation
                load_activations_and_indices_and_topk()

    def test_topk_larger_than_experts(self, temp_dir):
        """Test handling of topk larger than number of experts."""
        data = {
            "topk": 10,  # Larger than number of experts (4)
            "router_logits": th.randn(3, 2, 4),
        }
        th.save(data, temp_dir / "0.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            with pytest.raises(Exception):  # Should fail during topk operation
                load_activations_and_indices_and_topk()


class TestActivationLoadingIntegration:
    """Integration tests for activation loading."""

    def test_realistic_data_loading(self, temp_dir):
        """Test loading with realistic data sizes and patterns."""
        # Create realistic test data
        num_files = 5
        tokens_per_file = 1000
        num_layers = 12
        num_experts = 64
        topk = 8

        for i in range(num_files):
            data = {
                "topk": topk,
                "router_logits": th.randn(tokens_per_file, num_layers, num_experts),
            }
            th.save(data, temp_dir / f"{i}.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            activated_experts, activated_indices, loaded_topk = (
                load_activations_and_indices_and_topk()
            )

        total_tokens = num_files * tokens_per_file
        assert_tensor_shape_and_type(
            activated_experts, (total_tokens, num_layers, num_experts), th.bool
        )
        assert_tensor_shape_and_type(
            activated_indices, (total_tokens, num_layers, topk)
        )
        assert loaded_topk == topk

        # Check that exactly topk experts are active per layer per token
        active_counts = activated_experts.sum(dim=2)  # Sum over experts
        assert th.all(active_counts == topk)

    def test_memory_efficiency(self, temp_dir):
        """Test that loading doesn't cause memory issues with multiple files."""
        # Create multiple small files to test concatenation
        num_files = 10

        for i in range(num_files):
            data = {"topk": 2, "router_logits": th.randn(100, 6, 16)}
            th.save(data, temp_dir / f"{i}.pt")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            # Should complete without memory errors
            activated_experts, _, _ = load_activations_and_indices_and_topk()

            # Verify final shape
            assert_tensor_shape_and_type(activated_experts, (1000, 6, 16), th.bool)

    def test_different_layer_expert_combinations(self, temp_dir):
        """Test loading with different layer and expert combinations."""
        test_cases = [
            (1, 4),  # Single layer, few experts
            (8, 32),  # Multiple layers, many experts
            (24, 128),  # Many layers, very many experts
        ]

        for num_layers, num_experts in test_cases:
            # Clear directory
            for file in temp_dir.glob("*.pt"):
                file.unlink()

            data = {
                "topk": min(4, num_experts),  # Ensure topk <= num_experts
                "router_logits": th.randn(50, num_layers, num_experts),
            }
            th.save(data, temp_dir / "0.pt")

            with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
                activated_experts, _, _ = load_activations_and_indices_and_topk()
                assert_tensor_shape_and_type(
                    activated_experts, (50, num_layers, num_experts), th.bool
                )
