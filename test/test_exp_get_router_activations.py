"""Tests for exp.get_router_activations module."""

import os
from unittest.mock import MagicMock, patch
import warnings

import pytest
import torch as th
import yaml

from exp.get_router_activations import (
    CONFIG_FILENAME,
    ROUTER_LOGITS_DIRNAME,
    get_experiment_name,
    get_router_activations,
    save_config,
    verify_config,
)


class TestExperimentManagement:
    """Test experiment management functionality."""

    def test_get_experiment_name_basic(self):
        """Test basic experiment name generation."""
        name = get_experiment_name("gpt", "lmsys", batch_size=4, tokens_per_file=2000)
        assert name == "gpt_lmsys_batch_size=4_tokens_per_file=2000"

    def test_get_experiment_name_filters_keys(self):
        """Test that certain keys are filtered from experiment name."""
        with warnings.catch_warnings(record=True) as w:
            name = get_experiment_name(
                "gpt", "lmsys", device="cpu", resume=True, _hidden=123
            )
            assert name == "gpt_lmsys"
            assert len(w) == 1
            assert "excluded from the experiment name" in str(w[0].message)
            assert "device" in str(w[0].message)
            assert "resume" in str(w[0].message)
            assert "_hidden" in str(w[0].message)

    def test_save_and_verify_config(self, temp_dir):
        """Test saving and verifying configuration."""
        experiment_dir = temp_dir / "test_experiment"
        os.makedirs(experiment_dir, exist_ok=True)

        config = {"model_name": "gpt", "dataset_name": "lmsys", "batch_size": 4}

        # Test saving config
        save_config(config, str(experiment_dir))
        config_path = experiment_dir / CONFIG_FILENAME
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = yaml.safe_load(f)
        assert saved_config == config

        # Test verifying matching config
        verify_config(config, str(experiment_dir))  # Should not raise

        # Test verifying mismatched config
        mismatched_config = config.copy()
        mismatched_config["batch_size"] = 8
        with pytest.raises(ValueError) as excinfo:
            verify_config(mismatched_config, str(experiment_dir))
        assert "Configuration mismatch" in str(excinfo.value)
        assert "batch_size" in str(excinfo.value)

    def test_save_router_logits(self, temp_dir, monkeypatch):
        """Test saving router logits."""
        from exp.get_router_activations import save_router_logits

        # Set up patches
        monkeypatch.setattr("exp.get_router_activations.OUTPUT_DIR", str(temp_dir))

        # Create test data
        router_logit_collection = [th.tensor([[0.1, 0.2], [0.3, 0.4]])]
        tokenized_batch_collection = [[["token1", "token2"], ["token3", "token4"]]]
        top_k = 2
        file_idx = 1
        experiment_name = "test_experiment"

        # Create experiment directory
        experiment_dir = os.path.join(str(temp_dir), experiment_name)
        router_logits_dir = os.path.join(experiment_dir, ROUTER_LOGITS_DIRNAME)
        os.makedirs(router_logits_dir, exist_ok=True)

        # Mock torch.cat and torch.save
        with (
            patch("torch.cat", return_value=th.tensor([[0.1, 0.2], [0.3, 0.4]])),
            patch("torch.save") as mock_save,
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            # Call the function
            save_router_logits(
                router_logit_collection,
                tokenized_batch_collection,
                top_k,
                file_idx,
                experiment_name,
            )

            # Check that torch.save was called with the right arguments
            mock_save.assert_called_once()

            # Get the arguments passed to torch.save
            args, _ = mock_save.call_args
            output_dict, output_path = args

            # Check the output dictionary
            assert "topk" in output_dict
            assert output_dict["topk"] == top_k
            assert "router_logits" in output_dict
            assert "tokens" in output_dict
            assert output_dict["tokens"] == tokenized_batch_collection

            # Check the output path
            expected_path = os.path.join(router_logits_dir, f"{file_idx}.pt")
            assert output_path == expected_path


class TestGetRouterActivations:
    """Test get_router_activations function."""

    @pytest.mark.skip(
        reason="Integration test requiring complex mocking of StandardizedTransformer"
    )
    def test_get_router_activations_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of get_router_activations."""
        # Set up patches
        monkeypatch.setattr("exp.get_router_activations.OUTPUT_DIR", str(temp_dir))

        # Mock dependencies
        mock_dataset_fn = MagicMock(return_value=["text1", "text2"])

        # Create a mock model
        mock_model_config = MagicMock()
        mock_model_config.hf_name = "test_model"

        # Create a mock StandardizedTransformer instance
        mock_transformer = MagicMock()
        mock_transformer.layers_with_routers = [0, 1]
        mock_transformer.router_probabilities.get_top_k.return_value = 2

        # Create a context manager for inference_mode
        mock_inference_mode = MagicMock()
        mock_inference_mode.__enter__ = MagicMock()
        mock_inference_mode.__exit__ = MagicMock()

        with (
            patch(
                "exp.get_router_activations.MODELS",
                {"test_model": mock_model_config},
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
                return_value=mock_transformer,
            ),
            patch(
                "exp.get_router_activations.process_batch",
                return_value=(th.rand(2, 2, 4), [["token1", "token2"], ["token3"]]),
            ),
            patch(
                "exp.get_router_activations.save_router_logits",
            ) as mock_save_router_logits,
            patch(
                "exp.get_router_activations.save_config",
            ) as mock_save_config,
            patch(
                "exp.get_router_activations.verify_config",
            ) as mock_verify_config,
            patch(
                "os.makedirs",
            ) as mock_makedirs,
            patch(
                "itertools.batched",
                return_value=[["text1", "text2"]],
            ),
            patch(
                "tqdm.tqdm",
                side_effect=lambda x, **kwargs: x,
            ),
            patch(
                "gc.collect",
            ),
            patch(
                "th.cuda.empty_cache",
            ),
            patch(
                "th.cuda.is_available",
                return_value=False,
            ),
            patch(
                "th.inference_mode",
                return_value=mock_inference_mode,
            ),
        ):
            # Call the function
            get_router_activations(
                model_name="test_model",
                dataset_name="test_dataset",
                batch_size=2,
                device="cpu",
                tokens_per_file=10,
            )

            # Verify that the experiment directories were created
            expected_exp_name = (
                "test_model_test_dataset_batch_size=2_tokens_per_file=10"
            )
            expected_exp_dir = os.path.join(str(temp_dir), expected_exp_name)
            expected_router_logits_dir = os.path.join(
                expected_exp_dir, ROUTER_LOGITS_DIRNAME
            )

            # Check that makedirs was called with the right paths
            mock_makedirs.assert_any_call(expected_exp_dir, exist_ok=True)
            mock_makedirs.assert_any_call(expected_router_logits_dir, exist_ok=True)

            # Check that config was saved and verified
            mock_save_config.assert_called_once()
            mock_verify_config.assert_called_once()

            # Check that save_router_logits was called
            mock_save_router_logits.assert_called()

    @pytest.mark.skip(
        reason="Integration test requiring complex mocking of StandardizedTransformer"
    )
    def test_get_router_activations_with_experiment_name(self, temp_dir, monkeypatch):
        """Test get_router_activations with custom experiment name."""
        # Set up patches
        monkeypatch.setattr("exp.get_router_activations.OUTPUT_DIR", str(temp_dir))

        # Mock dependencies
        mock_dataset_fn = MagicMock(return_value=["text1", "text2"])

        # Create a mock model
        mock_model_config = MagicMock()
        mock_model_config.hf_name = "test_model"

        # Create a mock StandardizedTransformer instance
        mock_transformer = MagicMock()
        mock_transformer.layers_with_routers = [0, 1]
        mock_transformer.router_probabilities.get_top_k.return_value = 2

        # Create a context manager for inference_mode
        mock_inference_mode = MagicMock()
        mock_inference_mode.__enter__ = MagicMock()
        mock_inference_mode.__exit__ = MagicMock()

        with (
            patch(
                "exp.get_router_activations.MODELS",
                {"test_model": mock_model_config},
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
                return_value=mock_transformer,
            ),
            patch(
                "exp.get_router_activations.process_batch",
                return_value=(th.rand(2, 2, 4), [["token1", "token2"], ["token3"]]),
            ),
            patch(
                "exp.get_router_activations.save_router_logits",
            ) as mock_save_router_logits,
            patch(
                "exp.get_router_activations.save_config",
            ) as mock_save_config,
            patch(
                "exp.get_router_activations.verify_config",
            ) as mock_verify_config,
            patch(
                "os.makedirs",
            ) as mock_makedirs,
            patch(
                "itertools.batched",
                return_value=[["text1", "text2"]],
            ),
            patch(
                "tqdm.tqdm",
                side_effect=lambda x, **kwargs: x,
            ),
            patch(
                "gc.collect",
            ),
            patch(
                "th.cuda.empty_cache",
            ),
            patch(
                "th.cuda.is_available",
                return_value=False,
            ),
            patch(
                "th.inference_mode",
                return_value=mock_inference_mode,
            ),
        ):
            # Call the function with custom experiment name
            custom_name = "custom_experiment"
            get_router_activations(
                model_name="test_model",
                dataset_name="test_dataset",
                batch_size=2,
                device="cpu",
                tokens_per_file=10,
                name=custom_name,
            )

            # Check that directories were created correctly
            expected_exp_dir = os.path.join(str(temp_dir), custom_name)
            expected_router_logits_dir = os.path.join(
                expected_exp_dir, ROUTER_LOGITS_DIRNAME
            )

            # Check that makedirs was called with the right paths
            mock_makedirs.assert_any_call(expected_exp_dir, exist_ok=True)
            mock_makedirs.assert_any_call(expected_router_logits_dir, exist_ok=True)

            # Check that config was saved and verified
            mock_save_config.assert_called_once()
            mock_verify_config.assert_called_once()

            # Check that save_router_logits was called
            mock_save_router_logits.assert_called()


class TestProcessBatch:
    """Test process_batch function."""

    @pytest.mark.skip(
        reason="Integration test requiring complex mocking of tensor operations"
    )
    def test_process_batch(self):
        """Test that process_batch correctly processes a batch."""
        # Import the function to test
        from exp.get_router_activations import process_batch

        # Create mock model
        mock_model = MagicMock()

        # Mock tokenizer
        encoded_batch = MagicMock()
        encoded_batch.attention_mask = th.tensor([[1, 1, 0], [1, 0, 0]])
        mock_model.tokenizer.return_value = encoded_batch
        mock_model.tokenizer.tokenize.side_effect = lambda text: [
            f"{text}_token1",
            f"{text}_token2",
        ]

        # Mock router outputs
        mock_router_output_0 = (MagicMock(), MagicMock())
        mock_router_output_1 = MagicMock()

        # Mock CPU and indexing operations
        cpu_result_0 = MagicMock()
        cpu_result_0.__getitem__.return_value = MagicMock(
            save=MagicMock(
                return_value=MagicMock(
                    clone=MagicMock(
                        return_value=MagicMock(
                            detach=MagicMock(
                                return_value=th.tensor([0.1, 0.2, 0.3, 0.4])
                            )
                        )
                    )
                )
            )
        )

        # Set up the CPU method for both router outputs
        mock_router_output_0[0].cpu.return_value = cpu_result_0
        mock_router_output_1.cpu.return_value = cpu_result_0

        # Set up the router outputs dictionary
        mock_model.routers_output = {0: mock_router_output_0, 1: mock_router_output_1}

        # Mock the trace context manager
        mock_tracer = MagicMock()
        mock_tracer.__enter__ = MagicMock(return_value=mock_tracer)
        mock_tracer.__exit__ = MagicMock(return_value=None)
        mock_tracer.stop = MagicMock()
        mock_model.trace.return_value = mock_tracer

        # Mock torch.stack
        stacked_tensor = th.rand(2, 2, 4)

        # Use MagicMock for clone method
        mock_clone_result = MagicMock()
        mock_clone_result.detach.return_value = stacked_tensor

        # Patch the clone method
        with (
            patch("torch.stack", return_value=stacked_tensor),
            patch.object(stacked_tensor, "clone", return_value=mock_clone_result),
        ):
            # Call the function
            router_logits, tokenized_batch = process_batch(
                ["text1", "text2"], mock_model, [0, 1]
            )

            # Check the results
            assert router_logits is stacked_tensor
            assert len(tokenized_batch) == 2
            assert tokenized_batch[0] == ["text1_token1", "text1_token2"]
            assert tokenized_batch[1] == ["text2_token1", "text2_token2"]

            # Verify that the tracer was stopped
            mock_tracer.stop.assert_called_once()
