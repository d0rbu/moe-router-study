import os

import pytest

from exp.get_router_activations import (
    CONFIG_FILENAME,
    get_experiment_name,
    save_config,
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
        config = {
            "model_name": "gpt2",
            "dataset_name": "lmsys",
            "tokens_per_file": 2000,
        }

        # Save the config
        save_config(config, experiment_dir)

        # Check that the config file exists
        config_path = os.path.join(experiment_dir, CONFIG_FILENAME)
        assert os.path.exists(config_path)

        # Verify the config (should not raise an exception)
        verify_config(config, experiment_dir)

        # Try to verify with a different config (should raise ValueError)
        different_config = config.copy()
        different_config["tokens_per_file"] = 4000
        with pytest.raises(ValueError):
            verify_config(different_config, experiment_dir)


# This class is a placeholder for future tests
class TestProcessBatch:
    """Test process_batch function."""

    def test_placeholder(self):
        """Placeholder test."""
        assert True
