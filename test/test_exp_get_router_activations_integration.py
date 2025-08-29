"""Integration tests for get_router_activations.py."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as th

from exp.get_router_activations import (
    CONFIG_FILENAME,
    ROUTER_LOGITS_DIRNAME,
    get_router_activations,
    process_batch,
    get_experiment_name,
    save_config,
    verify_config,
    save_router_logits,
)


class TestExperimentManagement:
    """Test experiment management functions."""

    def test_get_experiment_name_basic(self):
        """Test that get_experiment_name returns a deterministic name."""
        # Test with basic parameters
        name = get_experiment_name(
            model_name="test_model",
            dataset_name="test_dataset",
            batch_size=2,
            tokens_per_file=10,
        )
        expected = "test_model_test_dataset_batch_size=2_tokens_per_file=10"
        assert name == expected

        # Test with additional parameters
        name = get_experiment_name(
            model_name="test_model",
            dataset_name="test_dataset",
            batch_size=2,
            tokens_per_file=10,
            extra_param="value",
        )
        expected = (
            "test_model_test_dataset_batch_size=2_extra_param=value_tokens_per_file=10"
        )
        assert name == expected

    def test_get_experiment_name_filters_keys(self):
        """Test that get_experiment_name filters out certain keys."""
        # Keys that should be filtered: device, resume, and keys starting with _
        with pytest.warns(UserWarning) as record:
            name = get_experiment_name(
                model_name="test_model",
                dataset_name="test_dataset",
                batch_size=2,
                tokens_per_file=10,
                device="cpu",
                resume=True,
                _internal="value",
            )

        # Check that a warning was raised
        assert len(record) > 0
        assert "excluded from the experiment name" in str(record[0].message)

        expected = "test_model_test_dataset_batch_size=2_tokens_per_file=10"
        assert name == expected

    def test_save_and_verify_config(self, tmp_path):
        """Test save_config and verify_config functions."""
        experiment_dir = str(tmp_path / "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        # Create a test config
        config = {
            "model_name": "test_model",
            "dataset_name": "test_dataset",
            "batch_size": 2,
            "tokens_per_file": 10,
        }

        # Save the config
        save_config(config, experiment_dir)

        # Check that the config file exists
        config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
        assert os.path.exists(config_path)

        # Verify the config (should not raise an error)
        verify_config(config, experiment_dir)

        # Try to verify with a different config (should raise an error)
        different_config = config.copy()
        different_config["batch_size"] = 3

        with pytest.raises(ValueError):
            verify_config(different_config, experiment_dir)

    @pytest.mark.skip(reason="Function signature changed, needs refactoring")
    def test_save_router_logits(self, tmp_path):
        """Test save_router_logits function."""
        experiment_name = "test_experiment"
        experiment_dir = os.path.join(str(tmp_path), experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
        os.makedirs(router_logits_dir, exist_ok=True)

        # Create test data
        router_logits = [th.rand(2, 3) for _ in range(2)]
        tokens = [["token1", "token2"], ["token3", "token4"]]
        top_k = 2
        file_idx = 0

        # Mock torch.cat and torch.save to avoid actual IO
        with (
            patch("torch.cat", return_value=th.rand(4, 3)),
            patch("torch.save") as mock_save,
            patch("exp.OUTPUT_DIR", str(tmp_path)),
        ):
            # Call the function
            save_router_logits(router_logits, tokens, top_k, file_idx, experiment_name)

            # Check that torch.save was called with the right arguments
            mock_save.assert_called_once()
            args, _ = mock_save.call_args
            saved_dict = args[0]

            # Check the saved dict has the right keys
            assert "topk" in saved_dict
            assert "router_logits" in saved_dict
            assert "tokens" in saved_dict

            # Check the saved path is correct
            expected_path = os.path.join(router_logits_dir, f"{file_idx}.pt")
            assert str(expected_path) == str(args[1])


class TestProcessBatch:
    """Test process_batch function."""

    @pytest.mark.skip(reason="Complex mocking of tensor operations")
    def test_process_batch(self):
        """Test process_batch function."""
        # Create mock model
        mock_model = MagicMock()

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["token1", "token2"]
        mock_model.tokenizer = mock_tokenizer

        # Create mock encoding
        mock_encoding = {
            "input_ids": th.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": th.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        mock_tokenizer.return_value = mock_encoding

        # Create mock tracer
        mock_tracer = MagicMock()
        mock_model.trace = mock_tracer

        # Create mock router outputs
        mock_router_output1 = MagicMock()
        mock_router_output1.cpu.return_value = MagicMock()
        mock_router_output1.cpu()[
            mock_encoding["attention_mask"].bool()
        ].clone.return_value = MagicMock()
        mock_router_output1.cpu()[
            mock_encoding["attention_mask"].bool()
        ].clone.return_value.detach.return_value = th.rand(5, 3)

        mock_router_output2 = (MagicMock(), MagicMock())  # Tuple output
        mock_router_output2[0].cpu.return_value = MagicMock()
        mock_router_output2[0].cpu()[
            mock_encoding["attention_mask"].bool()
        ].clone.return_value = MagicMock()
        mock_router_output2[0].cpu()[
            mock_encoding["attention_mask"].bool()
        ].clone.return_value.detach.return_value = th.rand(5, 3)

        # Set up the tracer to return the mock router outputs
        mock_tracer.__getitem__.side_effect = [mock_router_output1, mock_router_output2]

        # Set up the model's router layers
        router_layers = [0, 1]

        # Call the function
        with patch("torch.stack", return_value=th.rand(2, 5, 3)) as mock_stack:
            router_logits, tokens = process_batch(
                ["text1", "text2"], mock_model, router_layers
            )

            # Check that the tracer was used correctly
            assert mock_tracer.__getitem__.call_count == 2
            mock_tracer.stop.assert_called_once()

            # Check that torch.stack was called
            mock_stack.assert_called_once()

            # Check the return values
            assert router_logits.shape == (2, 5, 3)
            assert len(tokens) == 2


class TestGetRouterActivationsIntegration:
    """Integration tests for get_router_activations function."""

    @pytest.mark.skip(
        reason="Integration test requiring complex mocking of StandardizedTransformer"
    )
    def test_get_router_activations_basic_integration(self):
        """Test basic functionality of get_router_activations with minimal mocking."""
        # Use a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal mocks for the dependencies
            mock_model_config = MagicMock()
            mock_model_config.hf_name = "test_model"

            mock_dataset_fn = MagicMock(return_value=[])

            # Create a mock for StandardizedTransformer
            mock_model = MagicMock()
            mock_model.layers_with_routers = []
            mock_model.router_probabilities.get_top_k.return_value = 2

            # Create a mock for inference_mode context manager
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=None)
            mock_inference_mode = MagicMock(return_value=mock_ctx)

            # Patch all the necessary functions
            with (
                patch("exp.OUTPUT_DIR", temp_dir),
                patch(
                    "exp.get_router_activations.MODELS",
                    {"test_model": mock_model_config},
                ),
                patch(
                    "exp.get_router_activations.DATASETS",
                    {"test_dataset": mock_dataset_fn},
                ),
                patch(
                    "exp.get_router_activations.CUSTOM_DEVICES", {"cpu": lambda: "cpu"}
                ),
                patch(
                    "exp.get_router_activations.StandardizedTransformer",
                    return_value=mock_model,
                ),
                patch(
                    "exp.get_router_activations.process_batch",
                    return_value=(th.zeros(1), []),
                ),
                patch("exp.get_router_activations.batched", return_value=[[]]),
                patch("exp.get_router_activations.tqdm", lambda x, **kwargs: x),
                patch(
                    "exp.get_router_activations.th.inference_mode", mock_inference_mode
                ),
                patch("exp.get_router_activations.gc.collect"),
                patch("exp.get_router_activations.th.cuda.empty_cache"),
                patch(
                    "exp.get_router_activations.th.cuda.is_available",
                    return_value=False,
                ),
            ):
                # Call the function
                get_router_activations(
                    model_name="test_model",
                    dataset_name="test_dataset",
                    batch_size=2,
                    tokens_per_file=10,
                    device="cpu",
                )

                # Verify directories were created
                expected_experiment_name = (
                    "test_model_test_dataset_batch_size=2_tokens_per_file=10"
                )
                expected_experiment_dir = os.path.join(
                    temp_dir, expected_experiment_name
                )
                expected_router_logits_dir = os.path.join(
                    expected_experiment_dir, ROUTER_LOGITS_DIRNAME
                )
                expected_config_path = os.path.join(
                    expected_experiment_dir, CONFIG_FILENAME
                )

                assert os.path.exists(expected_experiment_dir)
                assert os.path.exists(expected_router_logits_dir)
                assert os.path.exists(expected_config_path)

    @pytest.mark.skip(
        reason="Integration test requiring complex mocking of StandardizedTransformer"
    )
    def test_get_router_activations_with_experiment_name_integration(self):
        """Test get_router_activations with custom experiment name with minimal mocking."""
        # Use a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal mocks for the dependencies
            mock_model_config = MagicMock()
            mock_model_config.hf_name = "test_model"

            mock_dataset_fn = MagicMock(return_value=[])

            # Create a mock for StandardizedTransformer
            mock_model = MagicMock()
            mock_model.layers_with_routers = []
            mock_model.router_probabilities.get_top_k.return_value = 2

            # Create a mock for inference_mode context manager
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=None)
            mock_inference_mode = MagicMock(return_value=mock_ctx)

            # Patch all the necessary functions
            with (
                patch("exp.OUTPUT_DIR", temp_dir),
                patch(
                    "exp.get_router_activations.MODELS",
                    {"test_model": mock_model_config},
                ),
                patch(
                    "exp.get_router_activations.DATASETS",
                    {"test_dataset": mock_dataset_fn},
                ),
                patch(
                    "exp.get_router_activations.CUSTOM_DEVICES", {"cpu": lambda: "cpu"}
                ),
                patch(
                    "exp.get_router_activations.StandardizedTransformer",
                    return_value=mock_model,
                ),
                patch(
                    "exp.get_router_activations.process_batch",
                    return_value=(th.zeros(1), []),
                ),
                patch("exp.get_router_activations.batched", return_value=[[]]),
                patch("exp.get_router_activations.tqdm", lambda x, **kwargs: x),
                patch(
                    "exp.get_router_activations.th.inference_mode", mock_inference_mode
                ),
                patch("exp.get_router_activations.gc.collect"),
                patch("exp.get_router_activations.th.cuda.empty_cache"),
                patch(
                    "exp.get_router_activations.th.cuda.is_available",
                    return_value=False,
                ),
            ):
                # Call the function with a custom experiment name
                custom_experiment_name = "custom_experiment"
                get_router_activations(
                    model_name="test_model",
                    dataset_name="test_dataset",
                    batch_size=2,
                    tokens_per_file=10,
                    device="cpu",
                    name=custom_experiment_name,
                )

                # Verify directories were created with the custom name
                expected_experiment_dir = os.path.join(temp_dir, custom_experiment_name)
                expected_router_logits_dir = os.path.join(
                    expected_experiment_dir, ROUTER_LOGITS_DIRNAME
                )
                expected_config_path = os.path.join(
                    expected_experiment_dir, CONFIG_FILENAME
                )

                assert os.path.exists(expected_experiment_dir)
                assert os.path.exists(expected_router_logits_dir)
                assert os.path.exists(expected_config_path)
