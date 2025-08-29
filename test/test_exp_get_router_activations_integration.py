"""Integration tests for get_router_activations.py."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch as th

from exp.get_router_activations import (
    CONFIG_FILENAME,
    ROUTER_LOGITS_DIRNAME,
    get_router_activations,
)


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
