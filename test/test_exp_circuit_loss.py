"""Tests for exp.circuit_loss module."""

import pytest
import torch as th

from exp.circuit_loss import (
    circuit_loss,
    hard_circuit_score,
    max_iou_and_index,
    mean_max_iou,
    min_logit_loss,
    min_logit_loss_and_index,
)
from test.test_utils import (
    assert_tensor_close,
    assert_tensor_shape_and_type,
    assert_valid_iou_score,
    assert_valid_loss,
    create_sample_circuit_logits,
    create_sample_circuits,
    create_sample_router_data,
)


class TestMaxIouAndIndex:
    """Test max_iou_and_index function."""
    
    def test_basic_iou_calculation(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test basic IoU calculation."""
        max_iou, max_iou_idx = max_iou_and_index(sample_bool_tensor_3d, sample_circuits_tensor)
        
        # Check output shapes
        batch_size = sample_bool_tensor_3d.shape[0]
        num_circuits = sample_circuits_tensor.shape[0]
        
        assert_tensor_shape_and_type(max_iou, (batch_size,), th.float32)
        assert_tensor_shape_and_type(max_iou_idx, (batch_size,), th.long)
        
        # Check IoU values are valid
        assert_valid_iou_score(max_iou)
        
        # Check indices are valid
        assert th.all(max_iou_idx >= 0)
        assert th.all(max_iou_idx < num_circuits)
    
    def test_perfect_match_iou(self):
        """Test IoU calculation with perfect matches."""
        # Create data that perfectly matches one circuit
        data = th.zeros(1, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        data[0, 1, 1] = True
        
        circuits = th.zeros(2, 2, 3, dtype=th.bool)
        # Circuit 0: perfect match
        circuits[0, 0, 0] = True
        circuits[0, 1, 1] = True
        # Circuit 1: no match
        circuits[1, 0, 2] = True
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        # Should get perfect IoU (1.0) with circuit 0
        assert_tensor_close(max_iou, th.tensor([1.0]))
        assert_tensor_close(max_iou_idx, th.tensor([0]))
    
    def test_no_overlap_iou(self):
        """Test IoU calculation with no overlap."""
        # Create data and circuits with no overlap
        data = th.zeros(1, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        
        circuits = th.zeros(1, 2, 3, dtype=th.bool)
        circuits[0, 0, 1] = True  # Different expert
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        # Should get IoU of 0.0
        assert_tensor_close(max_iou, th.tensor([0.0]))
    
    def test_partial_overlap_iou(self):
        """Test IoU calculation with partial overlap."""
        # Create data with 2 active experts
        data = th.zeros(1, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        data[0, 0, 1] = True
        
        # Create circuit with 1 overlapping expert and 1 additional
        circuits = th.zeros(1, 2, 3, dtype=th.bool)
        circuits[0, 0, 0] = True  # Overlap
        circuits[0, 0, 2] = True  # Additional
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        # IoU = intersection / union = 1 / 3 = 0.333...
        expected_iou = 1.0 / 3.0
        assert_tensor_close(max_iou, th.tensor([expected_iou]), rtol=1e-4)
    
    def test_multiple_circuits_selection(self):
        """Test that the circuit with highest IoU is selected."""
        data = th.zeros(1, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        data[0, 0, 1] = True
        
        circuits = th.zeros(3, 2, 3, dtype=th.bool)
        # Circuit 0: IoU = 1/3
        circuits[0, 0, 0] = True
        circuits[0, 0, 2] = True
        # Circuit 1: IoU = 2/2 = 1.0 (perfect match)
        circuits[1, 0, 0] = True
        circuits[1, 0, 1] = True
        # Circuit 2: IoU = 1/4
        circuits[2, 0, 0] = True
        circuits[2, 1, 0] = True
        circuits[2, 1, 1] = True
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        # Should select circuit 1 with perfect IoU
        assert_tensor_close(max_iou, th.tensor([1.0]))
        assert_tensor_close(max_iou_idx, th.tensor([1]))
    
    def test_batch_processing(self):
        """Test IoU calculation with multiple batch items."""
        batch_size = 3
        data = th.zeros(batch_size, 2, 3, dtype=th.bool)
        # Different patterns for each batch item
        data[0, 0, 0] = True
        data[1, 0, 1] = True
        data[2, 0, 2] = True
        
        circuits = th.zeros(2, 2, 3, dtype=th.bool)
        circuits[0, 0, 0] = True  # Matches batch item 0
        circuits[1, 0, 1] = True  # Matches batch item 1
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        assert_tensor_shape_and_type(max_iou, (batch_size,))
        assert_tensor_shape_and_type(max_iou_idx, (batch_size,))
        
        # Batch items 0 and 1 should get perfect matches
        assert max_iou[0] == 1.0
        assert max_iou[1] == 1.0
        assert max_iou_idx[0] == 0
        assert max_iou_idx[1] == 1
    
    def test_input_validation(self):
        """Test input validation for max_iou_and_index."""
        # Test wrong data dimensions
        with pytest.raises(AssertionError, match="data must be of shape"):
            max_iou_and_index(th.zeros(2, 3), th.zeros(1, 2, 3, dtype=th.bool))
        
        # Test wrong circuits dimensions
        with pytest.raises(AssertionError, match="circuits must be of shape"):
            max_iou_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(2, 3))
        
        # Test mismatched layer dimensions
        with pytest.raises(AssertionError, match="circuits must have the same number of layers"):
            max_iou_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(1, 5, 4, dtype=th.bool))
        
        # Test mismatched expert dimensions
        with pytest.raises(AssertionError, match="circuits must have the same number of experts"):
            max_iou_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(1, 3, 6, dtype=th.bool))
        
        # Test wrong circuits dtype
        with pytest.raises(AssertionError, match="circuits must be a boolean tensor"):
            max_iou_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(1, 3, 4, dtype=th.float32))
    
    def test_extra_dimensions(self):
        """Test max_iou_and_index with extra dimensions in circuits."""
        data = th.zeros(2, 3, 4, dtype=th.bool)
        data[0, 0, 0] = True
        data[1, 1, 1] = True
        
        # Add extra dimension to circuits
        circuits = th.zeros(5, 2, 3, 4, dtype=th.bool)
        circuits[0, 0, 0, 0] = True  # Matches data[0]
        circuits[0, 1, 1, 1] = True  # Matches data[1]
        
        max_iou, max_iou_idx = max_iou_and_index(data, circuits)
        
        # Should handle extra dimensions correctly
        assert_tensor_shape_and_type(max_iou, (5, 2))
        assert_tensor_shape_and_type(max_iou_idx, (5, 2))


class TestMeanMaxIou:
    """Test mean_max_iou function."""
    
    def test_mean_calculation(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test mean IoU calculation."""
        mean_iou = mean_max_iou(sample_bool_tensor_3d, sample_circuits_tensor)
        
        # Should return scalar
        assert_tensor_shape_and_type(mean_iou, ())
        assert_valid_iou_score(mean_iou.unsqueeze(0))
    
    def test_mean_consistency_with_max_iou(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test that mean_max_iou is consistent with max_iou_and_index."""
        max_iou, _ = max_iou_and_index(sample_bool_tensor_3d, sample_circuits_tensor)
        mean_iou = mean_max_iou(sample_bool_tensor_3d, sample_circuits_tensor)
        
        expected_mean = max_iou.mean()
        assert_tensor_close(mean_iou, expected_mean)
    
    def test_perfect_match_mean(self):
        """Test mean IoU with perfect matches."""
        # Create data that perfectly matches circuits
        data = th.zeros(2, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        data[1, 0, 1] = True
        
        circuits = th.zeros(2, 2, 3, dtype=th.bool)
        circuits[0, 0, 0] = True  # Perfect match for data[0]
        circuits[1, 0, 1] = True  # Perfect match for data[1]
        
        mean_iou = mean_max_iou(data, circuits)
        
        # Should get perfect mean IoU
        assert_tensor_close(mean_iou, th.tensor(1.0))


class TestHardCircuitScore:
    """Test hard_circuit_score function."""
    
    def test_basic_score_calculation(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test basic circuit score calculation."""
        score = hard_circuit_score(sample_bool_tensor_3d, sample_circuits_tensor)
        
        # Should return scalar
        assert_tensor_shape_and_type(score, ())
        assert th.isfinite(score)
    
    def test_score_components(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test that score correctly combines IoU and complexity."""
        complexity_coeff = 0.5
        complexity_power = 1.0
        
        score = hard_circuit_score(
            sample_bool_tensor_3d, 
            sample_circuits_tensor,
            complexity_coefficient=complexity_coeff,
            complexity_power=complexity_power
        )
        
        # Calculate components separately
        iou = mean_max_iou(sample_bool_tensor_3d, sample_circuits_tensor)
        complexity = sample_circuits_tensor.sum(dim=(-3, -2, -1)).float()
        expected_score = iou - complexity_coeff * complexity.pow(complexity_power)
        
        assert_tensor_close(score, expected_score)
    
    def test_complexity_coefficient_effect(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test effect of complexity coefficient on score."""
        score_low = hard_circuit_score(
            sample_bool_tensor_3d, sample_circuits_tensor, complexity_coefficient=0.1
        )
        score_high = hard_circuit_score(
            sample_bool_tensor_3d, sample_circuits_tensor, complexity_coefficient=1.0
        )
        
        # Higher complexity coefficient should give lower score (more penalty)
        assert score_low > score_high
    
    def test_complexity_power_effect(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test effect of complexity power on score."""
        score_linear = hard_circuit_score(
            sample_bool_tensor_3d, sample_circuits_tensor, 
            complexity_coefficient=0.5, complexity_power=1.0
        )
        score_quadratic = hard_circuit_score(
            sample_bool_tensor_3d, sample_circuits_tensor,
            complexity_coefficient=0.5, complexity_power=2.0
        )
        
        # Quadratic penalty should be more severe for complex circuits
        complexity = sample_circuits_tensor.sum()
        if complexity > 1:
            assert score_quadratic < score_linear
    
    def test_zero_complexity_coefficient(self, sample_bool_tensor_3d, sample_circuits_tensor):
        """Test score with zero complexity coefficient."""
        score = hard_circuit_score(
            sample_bool_tensor_3d, sample_circuits_tensor, complexity_coefficient=0.0
        )
        iou = mean_max_iou(sample_bool_tensor_3d, sample_circuits_tensor)
        
        # Should equal pure IoU
        assert_tensor_close(score, iou)


class TestMinLogitLossAndIndex:
    """Test min_logit_loss_and_index function."""
    
    def test_basic_loss_calculation(self, sample_bool_tensor_3d):
        """Test basic logit loss calculation."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        
        loss, indices = min_logit_loss_and_index(sample_bool_tensor_3d, circuits_logits)
        
        batch_size = sample_bool_tensor_3d.shape[0]
        num_circuits = circuits_logits.shape[0]
        
        assert_tensor_shape_and_type(loss, (batch_size,))
        assert_tensor_shape_and_type(indices, (batch_size,))
        
        assert_valid_loss(loss)
        assert th.all(indices >= 0)
        assert th.all(indices < num_circuits)
    
    def test_loss_with_perfect_circuit(self):
        """Test loss calculation with perfect circuit match."""
        # Create data
        data = th.zeros(1, 2, 3, dtype=th.bool)
        data[0, 0, 0] = True
        data[0, 1, 1] = True
        
        # Create circuit logits where one circuit perfectly matches
        circuits_logits = th.full((2, 2, 3), -10.0)  # Very negative (sigmoid ≈ 0)
        # Circuit 0: perfect match with high logits
        circuits_logits[0, 0, 0] = 10.0  # High logit (sigmoid ≈ 1)
        circuits_logits[0, 1, 1] = 10.0  # High logit (sigmoid ≈ 1)
        
        loss, indices = min_logit_loss_and_index(data, circuits_logits)
        
        # Should select circuit 0 and have low loss
        assert indices[0] == 0
        assert loss[0] < 1.0  # Should be relatively low
    
    def test_input_validation_logit_loss(self):
        """Test input validation for min_logit_loss_and_index."""
        # Test wrong data dimensions
        with pytest.raises(AssertionError, match="data must be of shape"):
            min_logit_loss_and_index(th.zeros(2, 3), th.zeros(1, 2, 3))
        
        # Test wrong circuits_logits dimensions
        with pytest.raises(AssertionError, match="circuits_logits must be of shape"):
            min_logit_loss_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(2, 3))
        
        # Test mismatched dimensions
        with pytest.raises(AssertionError, match="circuits_logits must have the same number of layers"):
            min_logit_loss_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(1, 5, 4))
        
        # Test wrong dtype
        with pytest.raises(AssertionError, match="circuits_logits must be a float32 tensor"):
            min_logit_loss_and_index(th.zeros(2, 3, 4, dtype=th.bool), th.zeros(1, 3, 4, dtype=th.int32))
    
    def test_distance_calculation(self):
        """Test that distance calculation works correctly."""
        # Create simple test case
        data = th.zeros(1, 1, 2, dtype=th.bool)
        data[0, 0, 0] = True  # Only first expert active
        
        circuits_logits = th.zeros(2, 1, 2)
        # Circuit 0: matches data exactly
        circuits_logits[0, 0, 0] = 10.0  # sigmoid ≈ 1
        circuits_logits[0, 0, 1] = -10.0  # sigmoid ≈ 0
        # Circuit 1: opposite of data
        circuits_logits[1, 0, 0] = -10.0  # sigmoid ≈ 0
        circuits_logits[1, 0, 1] = 10.0  # sigmoid ≈ 1
        
        loss, indices = min_logit_loss_and_index(data, circuits_logits)
        
        # Should select circuit 0 (closer match)
        assert indices[0] == 0


class TestMinLogitLoss:
    """Test min_logit_loss function."""
    
    def test_mean_calculation(self, sample_bool_tensor_3d):
        """Test mean logit loss calculation."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        
        loss = min_logit_loss(sample_bool_tensor_3d, circuits_logits)
        
        # Should return scalar
        assert_tensor_shape_and_type(loss, ())
        assert_valid_loss(loss.unsqueeze(0))
    
    def test_consistency_with_min_logit_loss_and_index(self, sample_bool_tensor_3d):
        """Test consistency with min_logit_loss_and_index."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        
        loss_individual, _ = min_logit_loss_and_index(sample_bool_tensor_3d, circuits_logits)
        loss_mean = min_logit_loss(sample_bool_tensor_3d, circuits_logits)
        
        expected_mean = loss_individual.mean()
        assert_tensor_close(loss_mean, expected_mean)


class TestCircuitLoss:
    """Test circuit_loss function."""
    
    def test_basic_circuit_loss(self, sample_bool_tensor_3d):
        """Test basic circuit loss calculation."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        
        total_loss, faithfulness_loss, complexity = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k
        )
        
        # Check output shapes
        assert_tensor_shape_and_type(total_loss, ())
        assert_tensor_shape_and_type(faithfulness_loss, ())
        assert_tensor_shape_and_type(complexity, ())
        
        # Check values are valid
        assert_valid_loss(total_loss.unsqueeze(0))
        assert_valid_loss(faithfulness_loss.unsqueeze(0))
        assert th.isfinite(complexity)
        assert complexity >= 0
    
    def test_loss_components_relationship(self, sample_bool_tensor_3d):
        """Test relationship between loss components."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        complexity_importance = 0.3
        
        total_loss, faithfulness_loss, complexity = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k, 
            complexity_importance=complexity_importance
        )
        
        # Calculate expected total loss
        faithfulness_importance = 1.0 - complexity_importance
        expected_total = faithfulness_importance * faithfulness_loss + complexity_importance * complexity
        
        assert_tensor_close(total_loss, expected_total, rtol=1e-5)
    
    def test_complexity_importance_bounds(self, sample_bool_tensor_3d):
        """Test complexity importance bounds validation."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        
        # Test invalid complexity importance values
        with pytest.raises(AssertionError, match="complexity_importance must be between 0.0 and 1.0"):
            circuit_loss(sample_bool_tensor_3d, circuits_logits, top_k, complexity_importance=-0.1)
        
        with pytest.raises(AssertionError, match="complexity_importance must be between 0.0 and 1.0"):
            circuit_loss(sample_bool_tensor_3d, circuits_logits, top_k, complexity_importance=1.1)
    
    def test_complexity_calculation(self):
        """Test complexity calculation logic."""
        # Create simple test case
        circuits_logits = th.zeros(2, 3, 4)  # 2 circuits, 3 layers, 4 experts
        circuits_logits[0] = 1.0  # Circuit 0: all positive
        circuits_logits[1, 0, 0] = 2.0  # Circuit 1: only one high value
        
        data = th.zeros(1, 3, 4, dtype=th.bool)
        top_k = 2
        
        _, _, complexity = circuit_loss(data, circuits_logits, top_k)
        
        # Complexity should be positive and finite
        assert th.isfinite(complexity)
        assert complexity > 0
    
    def test_pure_faithfulness_loss(self, sample_bool_tensor_3d):
        """Test circuit loss with pure faithfulness (no complexity penalty)."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        
        total_loss, faithfulness_loss, _ = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k, complexity_importance=0.0
        )
        
        # Total loss should equal faithfulness loss
        assert_tensor_close(total_loss, faithfulness_loss)
    
    def test_pure_complexity_loss(self, sample_bool_tensor_3d):
        """Test circuit loss with pure complexity penalty (no faithfulness)."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        
        total_loss, _, complexity = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k, complexity_importance=1.0
        )
        
        # Total loss should equal complexity
        assert_tensor_close(total_loss, complexity)
    
    def test_complexity_power_effect(self, sample_bool_tensor_3d):
        """Test effect of complexity power on loss."""
        circuits_logits = create_sample_circuit_logits(3, 3, 4)
        top_k = 2
        
        _, _, complexity_linear = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k, 
            complexity_importance=1.0, complexity_power=1.0
        )
        
        _, _, complexity_quadratic = circuit_loss(
            sample_bool_tensor_3d, circuits_logits, top_k,
            complexity_importance=1.0, complexity_power=2.0
        )
        
        # Quadratic should be different from linear (unless complexity is 0 or 1)
        if complexity_linear > 1.0:
            assert complexity_quadratic > complexity_linear
        elif 0 < complexity_linear < 1.0:
            assert complexity_quadratic < complexity_linear
    
    def test_topk_normalization(self):
        """Test that top-k normalization works correctly."""
        # Create circuits with known sigmoid values
        circuits_logits = th.zeros(1, 2, 4)
        circuits_logits[0, 0, :] = 10.0  # All experts highly active in layer 0
        circuits_logits[0, 1, 0] = 10.0  # Only first expert active in layer 1
        
        data = th.zeros(1, 2, 4, dtype=th.bool)
        top_k = 2
        
        _, _, complexity = circuit_loss(data, circuits_logits, top_k)
        
        # Complexity should account for top-k normalization
        assert th.isfinite(complexity)
        assert complexity > 0


class TestCircuitLossIntegration:
    """Integration tests for circuit loss functions."""
    
    def test_loss_functions_consistency(self):
        """Test consistency between different loss functions."""
        # Create test data
        data = create_sample_router_data(batch_size=4, num_layers=3, num_experts=8)
        circuits = create_sample_circuits(num_circuits=5, num_layers=3, num_experts=8)
        circuits_logits = create_sample_circuit_logits(num_circuits=5, num_layers=3, num_experts=8)
        
        # Test that all functions work together
        max_iou, _ = max_iou_and_index(data['activated_experts'], circuits)
        mean_iou = mean_max_iou(data['activated_experts'], circuits)
        hard_score = hard_circuit_score(data['activated_experts'], circuits)
        min_loss = min_logit_loss(data['activated_experts'], circuits_logits)
        total_loss, faith_loss, complexity = circuit_loss(
            data['activated_experts'], circuits_logits, data['topk']
        )
        
        # All should produce valid outputs
        assert_valid_iou_score(max_iou)
        assert_valid_iou_score(mean_iou.unsqueeze(0))
        assert th.isfinite(hard_score)
        assert_valid_loss(min_loss.unsqueeze(0))
        assert_valid_loss(total_loss.unsqueeze(0))
        assert_valid_loss(faith_loss.unsqueeze(0))
        assert th.isfinite(complexity)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through loss functions."""
        # Create data requiring gradients
        data = th.zeros(2, 3, 4, dtype=th.bool)
        data[0, 0, 0] = True
        data[1, 1, 1] = True
        
        circuits_logits = th.randn(3, 3, 4, requires_grad=True)
        
        # Compute loss
        total_loss, _, _ = circuit_loss(data, circuits_logits, top_k=2)
        
        # Backpropagate
        total_loss.backward()
        
        # Check that gradients exist and are finite
        assert circuits_logits.grad is not None
        assert th.all(th.isfinite(circuits_logits.grad))
        assert not th.all(circuits_logits.grad == 0)  # Should have non-zero gradients

