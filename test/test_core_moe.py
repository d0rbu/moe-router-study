"""Tests for core.moe module."""

import pytest
import torch as th

from core.device import get_backend
from core.moe import convert_router_logits_to_paths
from test.utils import assert_tensor_shape, create_synthetic_router_logits


class TestConvertRouterLogitsToPaths:
    """Test convert_router_logits_to_paths function."""

    def test_basic_functionality(self):
        """Test basic functionality with known inputs."""
        # Create simple router logits
        router_logits = th.tensor(
            [
                [
                    [1.0, 2.0, 0.5, 3.0],  # Expert 3 and 1 should be top-2
                    [0.1, 4.0, 2.0, 1.0],
                ]  # Expert 1 and 2 should be top-2
            ]
        )  # Shape: (1, 2, 4)

        top_k = 2
        result = convert_router_logits_to_paths(router_logits, top_k)

        # Check output shape
        batch_size, seq_len, num_experts = router_logits.shape
        expected_shape = (batch_size, seq_len, num_experts)
        assert_tensor_shape(result, expected_shape)

        # Check that exactly top_k experts are selected per position
        for b in range(batch_size):
            for s in range(seq_len):
                selected_experts = result[b, s].sum().item()
                assert selected_experts == top_k, (
                    f"Expected {top_k} experts, got {selected_experts}"
                )

    def test_top_k_selection(self):
        """Test that correct top-k experts are selected."""
        # Create router logits where we know the top experts
        router_logits = th.tensor(
            [
                [[1.0, 5.0, 2.0, 3.0]]  # Expert 1 (5.0) and 3 (3.0) should be top-2
            ]
        )  # Shape: (1, 1, 4)

        top_k = 2
        result = convert_router_logits_to_paths(router_logits, top_k)

        # Check individual positions (result is already unflattened)
        selected_mask = result[0, 0]  # Shape: (4,)

        # Check that experts 1 and 3 are selected (indices with highest logits)
        assert selected_mask[1].item() == 1.0  # Expert 1 (highest logit: 5.0)
        assert selected_mask[3].item() == 1.0  # Expert 3 (second highest: 3.0)
        assert selected_mask[0].item() == 0.0  # Expert 0 (third highest: 1.0)
        assert selected_mask[2].item() == 0.0  # Expert 2 (lowest: 2.0)

    def test_different_top_k_values(self):
        """Test with different top_k values."""
        router_logits = create_synthetic_router_logits(
            batch_size=1, seq_len=1, num_experts=8, seed=42
        )

        for top_k in [1, 2, 4, 8]:
            result = convert_router_logits_to_paths(router_logits, top_k)

            # Check shape
            expected_shape = (1, 1, 8)  # (batch_size, seq_len, num_experts)
            assert_tensor_shape(result, expected_shape)

            # Check that exactly top_k experts are selected
            selected_count = result[0, 0].sum().item()
            assert selected_count == top_k

    def test_multiple_batch_and_sequence(self):
        """Test with multiple batch and sequence dimensions."""
        batch_size, seq_len, num_experts = 3, 5, 6
        router_logits = create_synthetic_router_logits(
            batch_size=batch_size, seq_len=seq_len, num_experts=num_experts, seed=123
        )

        top_k = 2
        result = convert_router_logits_to_paths(router_logits, top_k)

        # Check output shape
        expected_shape = (batch_size, seq_len, num_experts)
        assert_tensor_shape(result, expected_shape)

        # Check that each position has exactly top_k selected experts
        for b in range(batch_size):
            for s in range(seq_len):
                selected_count = result[b, s].sum().item()
                assert selected_count == top_k

    def test_edge_case_top_k_equals_num_experts(self):
        """Test when top_k equals the number of experts."""
        router_logits = create_synthetic_router_logits(
            batch_size=2, seq_len=3, num_experts=4, seed=456
        )

        top_k = 4  # Same as num_experts
        result = convert_router_logits_to_paths(router_logits, top_k)

        # All experts should be selected
        for b in range(2):
            for s in range(3):
                selected_count = result[b, s].sum().item()
                assert selected_count == 4
                # All values should be 1
                assert th.all(result[b, s] == 1.0)

    def test_edge_case_top_k_one(self):
        """Test when top_k is 1."""
        router_logits = create_synthetic_router_logits(
            batch_size=2, seq_len=2, num_experts=5, seed=789
        )

        top_k = 1
        result = convert_router_logits_to_paths(router_logits, top_k)

        # Only one expert should be selected per position
        for b in range(2):
            for s in range(2):
                selected_count = result[b, s].sum().item()
                assert selected_count == 1
                # Exactly one value should be 1, others should be 0
                assert th.sum(result[b, s] == 1.0).item() == 1
                assert th.sum(result[b, s] == 0.0).item() == 4

    def test_output_dtype_and_device(self):
        """Test that output has correct dtype and device."""
        router_logits = create_synthetic_router_logits(
            batch_size=1, seq_len=2, num_experts=4, seed=42
        )

        top_k = 2
        result = convert_router_logits_to_paths(router_logits, top_k)

        # Should have same dtype and device as input
        assert result.dtype == router_logits.dtype
        assert result.device == router_logits.device

    def test_gradient_flow(self):
        """Test that the operation works with gradient computation."""
        router_logits = create_synthetic_router_logits(
            batch_size=1, seq_len=1, num_experts=4, seed=42
        )
        router_logits.requires_grad_(True)

        top_k = 2
        result = convert_router_logits_to_paths(router_logits, top_k)

        # The operation involves topk and scatter which don't preserve gradients
        # in the traditional sense, but we can test that the computation graph
        # is properly constructed and the operation completes without error
        assert result.requires_grad is False  # scatter_ breaks gradient flow

        # Test that we can still compute with the result
        output = result.sum()
        assert (
            output.item() == top_k
        )  # Should equal top_k since we select top_k experts per position

    def test_deterministic_behavior(self):
        """Test that the function is deterministic for the same inputs."""
        router_logits = create_synthetic_router_logits(
            batch_size=2, seq_len=3, num_experts=6, seed=42
        )

        top_k = 3
        result1 = convert_router_logits_to_paths(router_logits, top_k)
        result2 = convert_router_logits_to_paths(router_logits, top_k)

        # Results should be identical
        assert th.allclose(result1, result2)

    def test_different_dtypes(self):
        """Test with different input dtypes."""
        for dtype in [th.float32, th.float16]:
            if dtype == th.float16 and not get_backend("cuda").is_available():
                # Skip float16 on CPU as it might not be supported
                continue

            router_logits = create_synthetic_router_logits(
                batch_size=1, seq_len=2, num_experts=4, seed=42
            ).to(dtype)

            top_k = 2
            result = convert_router_logits_to_paths(router_logits, top_k)

            # Output should have same dtype as input
            assert result.dtype == dtype

            # Check basic functionality still works
            for s in range(2):
                selected_count = result[0, s].sum().item()
                assert selected_count == top_k

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_experts,top_k",
        [
            (1, 1, 4, 2),
            (2, 3, 8, 3),
            (4, 2, 6, 1),
            (1, 5, 10, 5),
        ],
    )
    def test_parametrized_shapes(self, batch_size, seq_len, num_experts, top_k):
        """Test with various parameter combinations."""
        router_logits = create_synthetic_router_logits(
            batch_size=batch_size, seq_len=seq_len, num_experts=num_experts, seed=42
        )

        result = convert_router_logits_to_paths(router_logits, top_k)

        # Check output shape
        expected_shape = (batch_size, seq_len, num_experts)
        assert_tensor_shape(result, expected_shape)

        # Check that each position has exactly top_k selected experts
        for b in range(batch_size):
            for s in range(seq_len):
                selected_count = result[b, s].sum().item()
                assert selected_count == top_k

    def test_invalid_inputs(self):
        """Test behavior with invalid inputs."""
        router_logits = create_synthetic_router_logits(
            batch_size=1, seq_len=2, num_experts=4, seed=42
        )

        # top_k larger than num_experts should raise an error
        top_k = 10  # Larger than num_experts (4)
        with pytest.raises(RuntimeError, match="selected index k out of range"):
            convert_router_logits_to_paths(router_logits, top_k)

    def test_zero_logits(self):
        """Test with zero logits."""
        router_logits = th.zeros(1, 2, 4)
        top_k = 2

        result = convert_router_logits_to_paths(router_logits, top_k)

        # Should still select top_k experts (even if they're all zero)
        for s in range(2):
            selected_count = result[0, s].sum().item()
            assert selected_count == top_k
