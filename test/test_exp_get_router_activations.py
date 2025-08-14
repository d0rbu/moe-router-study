"""Tests for exp.get_router_activations module."""

from unittest.mock import MagicMock, patch

import pytest
import torch as th

from exp.get_router_activations import get_router_activations


class TestGetRouterActivations:
    """Test get_router_activations function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_get_router_activations_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of get_router_activations."""
        # Set up patches
        monkeypatch.setattr("exp.get_router_activations.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr(
            "exp.get_router_activations.ROUTER_LOGITS_DIR", str(temp_dir)
        )

        # Mock dependencies
        mock_dataset_fn = MagicMock(return_value=["text1", "text2"])
        mock_model = MagicMock()
        mock_model.layers_with_routers = [0, 1]
        mock_model.router_probabilities.get_top_k.return_value = 2

        with (
            patch(
                "exp.get_router_activations.MODELS",
                {"test_model": MagicMock(hf_name="test_model")},
            ),
            patch(
                "exp.get_router_activations.DATASETS",
                {"test_dataset": mock_dataset_fn},
            ),
            patch(
                "exp.get_router_activations.CUSTOM_DEVICES",
                {"cpu": lambda: "cpu"},
            ),
            patch(
                "exp.get_router_activations.StandardizedTransformer",
                return_value=mock_model,
            ),
            patch(
                "exp.get_router_activations.process_batch",
                return_value=(th.rand(2, 2, 4), [["token1", "token2"], ["token3"]]),
            ),
            patch(
                "exp.get_router_activations.save_router_logits",
            ),
        ):
            # Call the function
            get_router_activations(
                model_name="test_model",
                dataset="test_dataset",
                batch_size=2,
                device="cpu",
                tokens_per_file=10,
            )

            # Check that the model was created with the right parameters
            from exp.get_router_activations import StandardizedTransformer

            StandardizedTransformer.assert_called_once()


class TestProcessBatch:
    """Test process_batch function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_process_batch(self):
        """Test that process_batch correctly processes a batch."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.tokenizer.return_value = MagicMock(
            attention_mask=th.tensor([[1, 1, 0], [1, 0, 0]])
        )
        mock_model.tokenizer.tokenize.side_effect = lambda text: [
            f"{text}_token1",
            f"{text}_token2",
        ]

        # Create mock router outputs
        mock_router_outputs = {
            0: MagicMock(
                router_logits=th.tensor(
                    [
                        [
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [0.9, 1.0, 1.1, 1.2],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                )
            ),
            1: MagicMock(
                router_logits=th.tensor(
                    [
                        [
                            [1.1, 1.2, 1.3, 1.4],
                            [1.5, 1.6, 1.7, 1.8],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                        [
                            [1.9, 2.0, 2.1, 2.2],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ],
                    ]
                )
            ),
        }

        # Create mock tracer
        mock_tracer = MagicMock()
        mock_model.trace.return_value = mock_tracer
        mock_model.routers_output = mock_router_outputs

        # Call the function
        from exp.get_router_activations import process_batch

        router_logits, tokenized_batch = process_batch(
            ["text1", "text2"], mock_model, [0, 1]
        )

        # Check the result
        assert isinstance(router_logits, th.Tensor)
        assert isinstance(tokenized_batch, list)
        assert len(tokenized_batch) == 2
        assert tokenized_batch[0] == ["text1_token1", "text1_token2"]
        assert tokenized_batch[1] == ["text2_token1", "text2_token2"]
