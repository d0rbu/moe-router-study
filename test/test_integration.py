"""Integration tests for the MoE router study codebase."""

from unittest.mock import MagicMock, patch

import pytest
import torch as th

from test.test_utils import (
    assert_tensor_shape_and_type,
    create_sample_circuit_logits,
    create_sample_circuits,
    create_sample_router_data,
)


class TestDataToActivationsPipeline:
    """Test the pipeline from data loading to activation processing."""

    def test_dataset_to_router_activations_flow(self):
        """Use toy dataset to avoid external downloads/timeouts."""
        from core.data import toy_text

        text_column = toy_text()
        collected_texts = list(text_column)

        # Verify toy dataset contents
        assert collected_texts == [
            "Tiny sample 1",
            "Tiny sample 2",
            "Tiny sample 3",
            "Tiny sample 4",
        ]


class TestActivationToLossPipeline:
    """Test the pipeline from activations to loss calculations."""

    def test_activation_to_circuit_loss_flow(self, temp_dir):
        """Test flow from activation loading to circuit loss calculation."""
        # Create test activation files
        num_files = 3
        tokens_per_file = 50
        num_layers = 4
        num_experts = 16
        topk = 4

        for i in range(num_files):
            data = {
                "topk": topk,
                "router_logits": th.randn(tokens_per_file, num_layers, num_experts),
            }
            th.save(data, temp_dir / f"{i}.pt")

        # Load activations
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            from exp.activations import load_activations_and_topk

            activated_experts, loaded_topk = load_activations_and_topk()

        # Verify activation loading
        total_tokens = num_files * tokens_per_file
        assert_tensor_shape_and_type(
            activated_experts, (total_tokens, num_layers, num_experts), th.bool
        )
        assert loaded_topk == topk

        # Create circuits for loss calculation
        num_circuits = 8
        circuits_logits = create_sample_circuit_logits(
            num_circuits, num_layers, num_experts
        )

        # Calculate circuit loss
        from exp.circuit_loss import circuit_loss

        total_loss, faithfulness_loss, complexity = circuit_loss(
            activated_experts, circuits_logits, topk
        )

        # Verify loss calculation
        assert th.isfinite(total_loss)
        assert th.isfinite(faithfulness_loss)
        assert th.isfinite(complexity)
        assert total_loss >= 0
        assert faithfulness_loss >= 0
        assert complexity >= 0


class TestLossToOptimizationPipeline:
    """Test the pipeline from loss calculation to optimization."""

    def test_loss_to_wandb_logging_flow(self):
        """Test flow from loss calculation to wandb logging."""
        # Create sample data
        batch_size = 10
        num_layers = 3
        num_experts = 8
        topk = 2

        # Generate sample activations
        router_data = create_sample_router_data(
            batch_size, num_layers, num_experts, topk
        )
        activated_experts = router_data["activated_experts"]

        # Generate sample circuits
        num_circuits = 5
        circuits_logits = create_sample_circuit_logits(
            num_circuits, num_layers, num_experts
        )

        # Calculate losses
        from exp.circuit_loss import circuit_loss

        total_loss, faithfulness_loss, complexity = circuit_loss(
            activated_experts, circuits_logits, topk
        )

        # Create batch for logging
        batch = {
            "total_loss": total_loss.unsqueeze(0),
            "faithfulness_loss": faithfulness_loss.unsqueeze(0),
            "complexity": complexity.unsqueeze(0),
            "step": th.tensor([100]),
        }

        # Test batch expansion
        from exp.circuit_optimization import expand_batch

        expanded = expand_batch(batch)

        # Verify expansion
        assert len(expanded) == 1
        assert "total_loss" in expanded[0]
        assert "faithfulness_loss" in expanded[0]
        assert "complexity" in expanded[0]
        assert "step" in expanded[0]

        # Verify values are scalars
        assert isinstance(expanded[0]["total_loss"], float)
        assert isinstance(expanded[0]["faithfulness_loss"], float)
        assert isinstance(expanded[0]["complexity"], float)
        assert isinstance(expanded[0]["step"], int)


class TestVisualizationPipeline:
    """Test the pipeline from activations to visualizations."""

    def test_activation_to_pca_visualization_flow(self, temp_dir):
        """Test flow from activations to PCA visualization."""
        # Create test activation files
        batch_size = 100
        num_layers = 6
        num_experts = 32
        topk = 8

        data = {
            "topk": topk,
            "router_logits": th.randn(batch_size, num_layers, num_experts),
        }
        th.save(data, temp_dir / "0.pt")

        # Mock the visualization pipeline
        with (
            patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)),
            patch("torch_pca.PCA") as mock_pca_class,
            patch("matplotlib.pyplot.scatter") as mock_scatter,
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.close") as mock_close,
        ):
            # Mock PCA
            mock_pca = MagicMock()
            pca_result = th.randn(batch_size, 2)
            mock_pca.fit_transform.return_value = pca_result
            mock_pca_class.return_value = mock_pca

            # Run PCA visualization
            from viz.pca_viz import pca_figure

            pca_figure()

            # Verify the pipeline
            mock_pca_class.assert_called_once_with(n_components=2, svd_solver="full")

            # Check that data was properly processed
            fit_transform_call_args = mock_pca.fit_transform.call_args[0]
            input_data = fit_transform_call_args[0]

            # Should be flattened and converted to float
            assert input_data.shape == (batch_size, num_layers * num_experts)
            assert input_data.dtype == th.float32

            # Verify matplotlib calls
            mock_scatter.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_weights_to_router_spaces_flow(self, temp_dir):
        """Test flow from weight extraction to router space visualization."""
        # Mock the StandardizedTransformer class
        with (
            patch("viz.router_spaces.StandardizedTransformer") as mock_transformer_class,
            patch("matplotlib.pyplot.plot") as mock_plot,
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.close") as mock_close,
            patch("os.makedirs"),
        ):
            # Set up the mock transformer
            mock_transformer = MagicMock()
            mock_transformer_class.return_value = mock_transformer

            # Mock router layers
            router_layers = [0, 2, 4]
            mock_transformer.layers_with_routers = router_layers

            # Mock router weights and o_proj weights
            num_experts = 16
            hidden_dim = 512

            # Create mock routers and attentions
            mock_routers = {}
            mock_attentions = {}

            for layer_idx in router_layers:
                # Create mock router
                mock_router = MagicMock()
                mock_router.weight = MagicMock()
                mock_router.weight.detach.return_value.cpu.return_value = th.randn(num_experts, hidden_dim)
                mock_routers[layer_idx] = mock_router

                # Create mock attention
                mock_attention = MagicMock()
                mock_attention.o_proj = MagicMock()
                mock_attention.o_proj.weight = MagicMock()
                mock_attention.o_proj.weight.detach.return_value.cpu.return_value = th.randn(hidden_dim, hidden_dim)
                mock_attentions[layer_idx] = mock_attention

            # Assign mock routers and attentions to the transformer
            mock_transformer.routers = mock_routers
            mock_transformer.attentions = mock_attentions

            # Run router spaces visualization
            from viz.router_spaces import router_spaces

            router_spaces()

            # Verify that plots were created
            assert mock_plot.call_count > 0
            assert mock_savefig.call_count > 0
            assert mock_close.call_count > 0

            # Verify that the transformer was initialized correctly
            mock_transformer_class.assert_called_once()

            # Verify that router weights were accessed
            for layer_idx in router_layers:
                assert mock_routers[layer_idx].weight.detach.called
                assert mock_attentions[layer_idx].o_proj.weight.detach.called


class TestEndToEndDataFlow:
    """Test end-to-end data flow through the entire pipeline."""

    def test_complete_pipeline_simulation(self, temp_dir):
        """Test a complete pipeline simulation with mocked components."""
        # Step 1: Mock dataset loading
        with patch("core.data.load_dataset") as mock_load_dataset:
            # Set up the mock to raise an exception to trigger the fallback path
            mock_load_dataset.side_effect = Exception("Simulated dataset loading error")

            # Then patch the fallback function to return our test data
            mock_text_data = ["Text sample 1", "Text sample 2"]
            with patch("core.data.toy_text") as mock_toy_text:
                mock_toy_text.return_value = mock_text_data

                from core.data import DATASETS

                # Use toy dataset
                dataset_func = DATASETS["toy"]
                text_data = list(dataset_func())

                # Verify the mocks were used correctly
                mock_load_dataset.assert_called_once_with(
                    "toy", split="train", streaming=True
                )
                mock_toy_text.assert_called_once()
                assert text_data == mock_text_data

        # Step 2: Simulate router activation extraction
        batch_size = 20
        num_layers = 4
        num_experts = 12
        topk = 3

        # Create activation files
        for i in range(2):
            data = {
                "topk": topk,
                "router_logits": th.randn(batch_size // 2, num_layers, num_experts),
            }
            th.save(data, temp_dir / f"{i}.pt")

        # Step 3: Load activations
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            from exp.activations import load_activations_and_indices_and_topk

            activated_experts, activated_indices, loaded_topk = (
                load_activations_and_indices_and_topk()
            )

        assert_tensor_shape_and_type(
            activated_experts, (batch_size, num_layers, num_experts), th.bool
        )
        assert_tensor_shape_and_type(activated_indices, (batch_size, num_layers, topk))
        assert loaded_topk == topk

        # Step 4: Circuit optimization
        num_circuits = 6
        circuits_logits = create_sample_circuit_logits(
            num_circuits, num_layers, num_experts
        )

        from exp.circuit_loss import circuit_loss

        total_loss, faithfulness_loss, complexity = circuit_loss(
            activated_experts, circuits_logits, topk
        )

        # Verify losses are valid
        assert th.isfinite(total_loss)
        assert th.isfinite(faithfulness_loss)
        assert th.isfinite(complexity)

        # Step 5: Batch logging preparation
        batch = {
            "total_loss": total_loss.unsqueeze(0).repeat(3),
            "faithfulness_loss": faithfulness_loss.unsqueeze(0).repeat(3),
            "complexity": complexity.unsqueeze(0).repeat(3),
            "step": th.tensor([100, 101, 102]),
        }

        from exp.circuit_optimization import expand_batch

        expanded = expand_batch(batch)

        assert len(expanded) == 3
        for item in expanded:
            assert all(
                key in item
                for key in ["total_loss", "faithfulness_loss", "complexity", "step"]
            )

        # Step 6: Visualization
        with (
            patch("torch_pca.PCA") as mock_pca_class,
            patch("matplotlib.pyplot.scatter"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            mock_pca = MagicMock()
            mock_pca.fit_transform.return_value = th.randn(batch_size, 2)
            mock_pca_class.return_value = mock_pca

            # This would normally use the loaded activations
            # For testing, we'll just verify the PCA setup
            mock_pca_class.assert_not_called()  # Not called yet

            # Simulate PCA call
            from torch_pca import PCA

            _ = PCA(n_components=2, svd_solver="full")

            # Verify we can process the activations
            flattened_activations = activated_experts.view(batch_size, -1).float()
            assert flattened_activations.shape == (batch_size, num_layers * num_experts)


class TestErrorPropagation:
    """Test error propagation through the pipeline."""

    def test_activation_loading_error_propagation(self, temp_dir):
        """Test that activation loading errors propagate correctly."""
        # Create corrupted file
        corrupted_file = temp_dir / "0.pt"
        corrupted_file.write_text("corrupted data")

        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            from exp.activations import load_activations_and_topk

            with pytest.raises((RuntimeError, ValueError, OSError)):
                load_activations_and_topk()

    def test_circuit_loss_error_propagation(self):
        """Test that circuit loss calculation errors propagate correctly."""
        # Create mismatched data
        data = th.zeros(2, 3, 4, dtype=th.bool)
        circuits_logits = th.zeros(2, 5, 4)  # Wrong number of layers

        from exp.circuit_loss import circuit_loss

        with pytest.raises(AssertionError):
            circuit_loss(data, circuits_logits, topk=2)

    def test_visualization_error_propagation(self):
        """Test that visualization errors propagate correctly."""
        with patch("exp.activations.load_activations") as mock_load:
            mock_load.side_effect = Exception("Activation loading failed")

            from viz.pca_viz import pca_figure

            with pytest.raises(Exception, match="Activation loading failed"):
                pca_figure()


class TestDataConsistency:
    """Test data consistency across pipeline stages."""

    def test_topk_consistency(self, temp_dir):
        """Test that topk values remain consistent throughout pipeline."""
        topk = 5
        batch_size = 30
        num_layers = 3
        num_experts = 20

        # Create activation data with specific topk
        data = {
            "topk": topk,
            "router_logits": th.randn(batch_size, num_layers, num_experts),
        }
        th.save(data, temp_dir / "0.pt")

        # Load and verify topk consistency
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            from exp.activations import load_activations_and_topk

            activated_experts, loaded_topk = load_activations_and_topk()

        assert loaded_topk == topk

        # Verify that exactly topk experts are active per token per layer
        active_counts = activated_experts.sum(dim=2)  # Sum over experts
        assert th.all(active_counts == topk)

        # Use in circuit loss calculation
        circuits_logits = create_sample_circuit_logits(4, num_layers, num_experts)

        from exp.circuit_loss import circuit_loss

        total_loss, _, _ = circuit_loss(activated_experts, circuits_logits, topk)

        # Should complete without errors
        assert th.isfinite(total_loss)

    def test_tensor_shape_consistency(self, temp_dir):
        """Test that tensor shapes remain consistent throughout pipeline."""
        batch_size = 25
        num_layers = 4
        num_experts = 16
        topk = 4

        # Create activation data
        data = {
            "topk": topk,
            "router_logits": th.randn(batch_size, num_layers, num_experts),
        }
        th.save(data, temp_dir / "0.pt")

        # Load activations
        with patch("exp.activations.ROUTER_LOGITS_DIR", str(temp_dir)):
            from exp.activations import load_activations_and_indices_and_topk

            activated_experts, activated_indices, _ = (
                load_activations_and_indices_and_topk()
            )

        # Verify shapes
        assert_tensor_shape_and_type(
            activated_experts, (batch_size, num_layers, num_experts), th.bool
        )
        assert_tensor_shape_and_type(activated_indices, (batch_size, num_layers, topk))

        # Use in circuit calculations
        num_circuits = 8
        circuits = create_sample_circuits(num_circuits, num_layers, num_experts)
        circuits_logits = create_sample_circuit_logits(
            num_circuits, num_layers, num_experts
        )

        # Test IoU calculation
        from exp.circuit_loss import max_iou_and_index

        max_iou, max_iou_idx = max_iou_and_index(activated_experts, circuits)

        assert_tensor_shape_and_type(max_iou, (batch_size,))
        assert_tensor_shape_and_type(max_iou_idx, (batch_size,))

        # Test logit loss calculation
        from exp.circuit_loss import min_logit_loss_and_index

        loss, loss_idx = min_logit_loss_and_index(activated_experts, circuits_logits)

        assert_tensor_shape_and_type(loss, (batch_size,))
        assert_tensor_shape_and_type(loss_idx, (batch_size,))
