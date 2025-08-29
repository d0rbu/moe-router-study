import os
from unittest.mock import MagicMock, patch

import pytest
import torch as th

from exp.get_router_activations import (
    CONFIG_FILENAME,
    ROUTER_LOGITS_DIRNAME,
    get_experiment_name,
    process_batch,
    save_config,
    save_router_logits,
    verify_config,
)


class TestExperimentManagement:
    """Test experiment management functions."""

    def test_get_experiment_name_basic(self):
        """Test basic experiment name generation."""
        name = get_experiment_name(
            model_name="gpt2",
            dataset_name="lmsys",
            batch_size=4,
            tokens_per_file=2000,
        )
        assert name == "gpt2_lmsys_batch_size=4_tokens_per_file=2000"

    def test_get_experiment_name_filters_keys(self):
        """Test that certain keys are filtered out of the experiment name."""
        with pytest.warns(UserWarning) as record:
            name = get_experiment_name(
                model_name="gpt2",
                dataset_name="lmsys",
                batch_size=4,
                device="cuda",
                resume=True,
                _internal_param=123,
            )

        # Check that the warning was raised
        assert len(record) == 1
        assert "excluded from the experiment name" in str(record[0].message)

        # Check that the filtered keys are not in the name
        assert name == "gpt2_lmsys_batch_size=4"
        assert "device" not in name
        assert "resume" not in name
        assert "_internal_param" not in name

    def test_save_and_verify_config(self, tmp_path):
        """Test saving and verifying configuration."""
        experiment_dir = str(tmp_path)
        config = {"model_name": "gpt2", "dataset_name": "lmsys", "batch_size": 4}

        # Save the config
        save_config(config, experiment_dir)

        # Check that the config file exists
        config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
        assert os.path.exists(config_path)

        # Verify the config (should not raise an exception)
        verify_config(config, experiment_dir)

        # Try to verify with a different config (should raise ValueError)
        different_config = config.copy()
        different_config["batch_size"] = 8
        with pytest.raises(ValueError):
            verify_config(different_config, experiment_dir)

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
        with patch("torch.cat", return_value=th.rand(4, 3)) as mock_cat, \
             patch("torch.save") as mock_save, \
             patch("exp.get_router_activations.OUTPUT_DIR", str(tmp_path)), \
             patch("gc.collect"), \
             patch("torch.cuda.is_available", return_value=False):

            # Call the function
            save_router_logits(router_logits, tokens, top_k, file_idx, experiment_name)

            # Check that torch.cat was called with the router_logits
            mock_cat.assert_called_once()
            cat_args, cat_kwargs = mock_cat.call_args
            assert cat_args[0] == router_logits
            assert cat_kwargs["dim"] == 0

            # Check that torch.save was called with the right arguments
            mock_save.assert_called_once()
            args, _ = mock_save.call_args
            saved_dict = args[0]

            # Check the saved dict has the right keys and values
            assert saved_dict["topk"] == top_k
            assert "router_logits" in saved_dict
            assert saved_dict["tokens"] == tokens

            # Check the saved path is correct
            expected_path = os.path.join(router_logits_dir, f"{file_idx}.pt")
            assert str(args[1]) == str(expected_path)


class TestProcessBatch:
    """Test process_batch function."""

    def test_process_batch(self):
        """Test process_batch function."""
        # Create mock model
        mock_model = MagicMock()

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.side_effect = lambda text: [f"{text}_token1", f"{text}_token2"]
        mock_model.tokenizer = mock_tokenizer

        # Create mock encoding with attention_mask
        mock_encoding = MagicMock()
        mock_encoding.attention_mask = th.tensor([[1, 1, 1], [1, 1, 0]])
        mock_tokenizer.return_value = mock_encoding

        # Create mock tracer
        mock_tracer = MagicMock()
        mock_model.trace.return_value.__enter__.return_value = mock_tracer

        # Set up router layers
        router_layers = [0, 1]

        # Set up mock routers_output
        mock_model.routers_output = {
            0: th.rand(6, 3),  # Regular tensor case
            1: (th.rand(6, 3), th.rand(6, 2))  # Tuple case
        }

        # Create mock tensors for the CPU and save operations
        mock_cpu_tensor1 = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save.return_value = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save.return_value.clone = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save.return_value.clone.return_value = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save.return_value.clone.return_value.detach = MagicMock()
        mock_cpu_tensor1.__getitem__.return_value.save.return_value.clone.return_value.detach.return_value = th.rand(5, 3)

        mock_cpu_tensor2 = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save.return_value = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save.return_value.clone = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save.return_value.clone.return_value = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save.return_value.clone.return_value.detach = MagicMock()
        mock_cpu_tensor2.__getitem__.return_value.save.return_value.clone.return_value.detach.return_value = th.rand(5, 3)

        # Mock the CPU operation
        with patch.object(th.Tensor, "cpu", side_effect=[mock_cpu_tensor1, mock_cpu_tensor2]), \
             patch("torch.stack", return_value=th.rand(2, 5, 3)):

            # Call the function
            router_logits, tokens = process_batch(["text1", "text2"], mock_model, router_layers)

            # Check the results
            assert router_logits.shape == (2, 5, 3)
            assert tokens == [["text1_token1", "text1_token2"], ["text2_token1", "text2_token2"]]

            # Verify the tracer was used correctly
            mock_tracer.stop.assert_called_once()

