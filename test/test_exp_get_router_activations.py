"""Tests for exp.get_router_activations module."""

import os
from unittest.mock import MagicMock, call, patch

import pytest
import torch as th

from exp.get_router_activations import (
    get_router_activations,
    process_batch,
    save_router_logits,
)


class TestSaveRouterLogits:
    """Test save_router_logits function."""

    def test_save_router_logits(self, temp_dir, monkeypatch):
        """Test that router logits are saved correctly."""
        # Set up test data
        router_logit_collection = [th.randn(2, 3, 4), th.randn(3, 3, 4)]
        tokenized_batch_collection = [
            ["token1", "token2"],
            ["token3", "token4", "token5"],
        ]
        top_k = 2
        file_idx = 0

        # Set output directory to temp_dir
        monkeypatch.setattr(
            "exp.get_router_activations.ROUTER_LOGITS_DIR", str(temp_dir)
        )

        # Call the function
        save_router_logits(
            router_logit_collection, tokenized_batch_collection, top_k, file_idx
        )

        # Check that the file was created
        output_file = os.path.join(temp_dir, "0.pt")
        assert os.path.exists(output_file)

        # Load and verify the saved data
        saved_data = th.load(output_file)
        assert "topk" in saved_data
        assert "router_logits" in saved_data
        assert "tokens" in saved_data

        assert saved_data["topk"] == top_k
        assert th.equal(
            saved_data["router_logits"], th.cat(router_logit_collection, dim=0)
        )
        assert saved_data["tokens"] == tokenized_batch_collection


class TestProcessBatch:
    """Test process_batch function."""

    def test_process_batch(self):
        """Test that process_batch correctly processes a batch."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.tokenizer.return_value = MagicMock(
            attention_mask=th.tensor([[1, 1, 0], [1, 0, 0]])
        )
        mock_model.tokenizer.tokenize.side_effect = lambda text: [
            f"{text}_token_{i}" for i in range(3)
        ]

        # Create mock tracer context manager
        mock_tracer = MagicMock()
        mock_model.trace.return_value = mock_tracer
        mock_tracer.__enter__.return_value = mock_tracer

        # Set up router outputs with proper save method
        class MockTensor(th.Tensor):
            def save(self):
                return self

        # Create router logits tensors that have a save method
        router_logits_0 = th.randn(6, 4)  # 6 tokens (flattened), 4 experts
        router_logits_1 = th.randn(6, 4)

        # Create mock CPU method that returns a tensor with save method
        def mock_cpu_0():
            tensor = MockTensor(router_logits_0)
            # Annotate to make shadowing explicit
            tensor.save = lambda: tensor  # type: ignore
            return tensor

        def mock_cpu_1():
            tensor = MockTensor(router_logits_1)
            # Annotate to make shadowing explicit
            tensor.save = lambda: tensor  # type: ignore
            return tensor

        # Set up router outputs
        mock_model.routers_output = {
            0: MagicMock(cpu=mock_cpu_0),
            1: MagicMock(cpu=mock_cpu_1),
        }

        # Mock the attention mask indexing
        mock_attention_mask = th.tensor(
            [True, True, False, True, False, False]
        )  # 6 tokens (flattened)
        mock_model.tokenizer.return_value.attention_mask.bool.return_value.view.return_value = mock_attention_mask

        # Call the function
        batch = ["text1", "text2"]
        router_layers = [0, 1]

        with patch("torch.stack", return_value=th.randn(3, 2, 4)) as mock_stack:
            router_logits, tokenized_batch = process_batch(
                batch, mock_model, router_layers
            )

        # Check that the tokenizer was called correctly
        mock_model.tokenizer.assert_called_once_with(
            batch, padding=True, return_tensors="pt"
        )

        # Check that the tokenize method was called for each text
        assert mock_model.tokenizer.tokenize.call_count == 2
        mock_model.tokenizer.tokenize.assert_has_calls([call("text1"), call("text2")])

        # Check that stack was called with the router logits
        assert mock_stack.call_count == 1

        # Check the tokenized batch
        assert len(tokenized_batch) == 2
        assert len(tokenized_batch[0]) == 3
        assert len(tokenized_batch[1]) == 3


class TestGetRouterActivations:
    """Test get_router_activations function."""

    def test_get_router_activations_invalid_model(self):
        """Test get_router_activations with invalid model name."""
        with (
            patch("exp.get_router_activations.MODELS", {}),
            pytest.raises(ValueError, match="Model .* not found"),
        ):
            get_router_activations(model_name="nonexistent_model")

    def test_get_router_activations_invalid_dataset(self):
        """Test get_router_activations with invalid dataset name."""
        with (
            patch("exp.get_router_activations.MODELS", {"test_model": MagicMock()}),
            patch("exp.get_router_activations.DATASETS", {}),
            pytest.raises(ValueError, match="Dataset .* not found"),
        ):
            get_router_activations(
                model_name="test_model", dataset="nonexistent_dataset"
            )

    def test_get_router_activations_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of get_router_activations."""
        # Mock dependencies
        mock_model_config = MagicMock(hf_name="test/model")
        mock_models = {"test_model": mock_model_config}

        # Create a simple dataset function that returns 3 items
        def mock_dataset_fn():
            yield "text1"
            yield "text2"
            yield "text3"

        mock_datasets = {"test_dataset": mock_dataset_fn}

        # Mock StandardizedTransformer
        mock_transformer = MagicMock()
        mock_transformer.layers_with_routers = [0, 1]
        mock_transformer.router_probabilities.get_top_k.return_value = 2

        # Mock process_batch to return predictable outputs
        def mock_process_batch(batch, model, router_layers):
            # Return router logits with shape (len(batch), len(router_layers), 4)
            return th.randn(len(batch), len(router_layers), 4), [
                f"{text}_tokenized" for text in batch
            ]

        # Set up patches
        monkeypatch.setattr("exp.get_router_activations.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr(
            "exp.get_router_activations.ROUTER_LOGITS_DIR", str(temp_dir)
        )

        with (
            patch("exp.get_router_activations.MODELS", mock_models),
            patch("exp.get_router_activations.DATASETS", mock_datasets),
            patch("exp.get_router_activations.CUSTOM_DEVICES", {"cpu": lambda: "cpu"}),
            patch(
                "exp.get_router_activations.StandardizedTransformer",
                return_value=mock_transformer,
            ),
            patch(
                "exp.get_router_activations.process_batch",
                side_effect=mock_process_batch,
            ),
            patch("exp.get_router_activations.save_router_logits") as mock_save,
        ):
            # Call the function with small tokens_per_file to ensure saving
            get_router_activations(
                model_name="test_model",
                dataset="test_dataset",
                batch_size=2,
                device="cpu",
                tokens_per_file=2,
            )

            # Check that save_router_logits was called
            assert mock_save.call_count > 0

    def test_get_router_activations_with_custom_device(self):
        """Test get_router_activations with custom device mapping."""
        # Mock dependencies
        mock_model_config = MagicMock(hf_name="test/model")
        mock_models = {"test_model": mock_model_config}

        # Create a simple dataset function that returns nothing (to exit loop quickly)
        def mock_dataset_fn():
            return
            yield  # This is never reached

        mock_datasets = {"test_dataset": mock_dataset_fn}

        # Mock custom device function
        def mock_custom_device():
            return {"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"}

        mock_custom_devices = {"test_device": mock_custom_device}

        # Mock StandardizedTransformer
        mock_transformer = MagicMock()
        mock_transformer.layers_with_routers = []
        mock_transformer.router_probabilities.get_top_k.return_value = 2

        with (
            patch("exp.get_router_activations.MODELS", mock_models),
            patch("exp.get_router_activations.DATASETS", mock_datasets),
            patch("exp.get_router_activations.CUSTOM_DEVICES", mock_custom_devices),
            patch(
                "exp.get_router_activations.StandardizedTransformer"
            ) as mock_transformer_cls,
            patch("exp.get_router_activations.os.makedirs"),
        ):
            # Call the function
            get_router_activations(
                model_name="test_model",
                dataset="test_dataset",
                device="test_device",
            )

            # Check that StandardizedTransformer was called with the custom device map
            mock_transformer_cls.assert_called_once()
            _, kwargs = mock_transformer_cls.call_args
            assert kwargs["device_map"] == {
                "model.layers.0": "cuda:0",
                "model.layers.1": "cuda:1",
            }
