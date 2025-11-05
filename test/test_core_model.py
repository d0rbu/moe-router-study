"""Tests for core.model module."""

import re
from unittest.mock import patch

import pytest

from core.model import MODELS, Checkpoint, ModelConfig
from test.test_utils import mock_huggingface_repo_refs


class TestCheckpoint:
    """Test Checkpoint dataclass."""

    def test_checkpoint_creation(self, mock_model_config):
        """Test basic checkpoint creation."""
        checkpoint = Checkpoint(
            step=1000, num_tokens=1000000, model_config=mock_model_config
        )

        assert checkpoint.step == 1000
        assert checkpoint.num_tokens == 1000000
        assert checkpoint.model_config == mock_model_config

    def test_checkpoint_str_with_tokens(self):
        """Test checkpoint string representation with tokens."""
        config = ModelConfig(hf_name="test/model", revision_format="step{}-tokens{}B")
        checkpoint = Checkpoint(step=1000, num_tokens=5, model_config=config)

        assert str(checkpoint) == "step1000-tokens5B"

    def test_checkpoint_str_without_tokens(self):
        """Test checkpoint string representation without tokens."""
        config = ModelConfig(hf_name="test/model", revision_format="step{}")
        checkpoint = Checkpoint(step=1000, num_tokens=None, model_config=config)

        assert str(checkpoint) == "step1000"

    def test_checkpoint_str_no_format(self):
        """Test checkpoint string representation with no format."""
        config = ModelConfig(hf_name="test/model")
        checkpoint = Checkpoint(step=1000, num_tokens=1000000, model_config=config)

        # Should handle None revision_format gracefully
        result = str(checkpoint)
        assert isinstance(result, str)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_basic_model_config(self):
        """Test basic model configuration creation."""
        config = ModelConfig(hf_name="test/model", tokenizer_has_padding_token=False)

        assert config.hf_name == "test/model"
        assert config.tokenizer_has_padding_token is False
        assert config.branch_regex is None
        assert config.revision_format is None
        assert config.checkpoints == []

    @patch("huggingface_hub.list_repo_refs")
    def test_model_config_with_checkpoints(self, mock_list_refs):
        """Test model configuration with checkpoint parsing."""
        # Mock repository refs
        branches = [
            "step1000-tokens1B",
            "step2000-tokens2B",
            "step500-tokens500M",
            "main",  # Should be ignored
            "invalid-branch",  # Should be ignored
        ]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
        )

        # Should have parsed 2 checkpoints (500M doesn't match the B pattern)
        assert len(config.checkpoints) == 2

        # Checkpoints should be sorted by step
        assert config.checkpoints[0].step == 1000
        assert config.checkpoints[0].num_tokens == 1
        assert config.checkpoints[1].step == 2000
        assert config.checkpoints[1].num_tokens == 2

    @patch("huggingface_hub.list_repo_refs")
    def test_model_config_single_group_regex(self, mock_list_refs):
        """Test model configuration with single group regex."""
        branches = ["step1000", "step2000", "step500"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model", branch_regex=r"step(\d+)", revision_format="step{}"
        )

        assert len(config.checkpoints) == 3

        # Check that num_tokens is None for single group
        for checkpoint in config.checkpoints:
            assert checkpoint.num_tokens is None

        # Check sorting
        steps = [c.step for c in config.checkpoints]
        assert steps == [500, 1000, 2000]

    @patch("huggingface_hub.list_repo_refs")
    def test_model_config_no_matching_branches(self, mock_list_refs):
        """Test model configuration with no matching branches."""
        branches = ["main", "dev", "feature-branch"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model", branch_regex=r"step(\d+)", revision_format="step{}"
        )

        assert len(config.checkpoints) == 0

    @patch("huggingface_hub.list_repo_refs")
    def test_model_config_invalid_regex_groups(self, mock_list_refs):
        """Test model configuration with invalid regex groups."""
        branches = ["step1000-tokens1B-extra"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        with pytest.raises(ValueError, match="Unexpected number of groups"):
            ModelConfig(
                hf_name="test/model",
                branch_regex=r"step(\d+)-tokens(\d+)B-(\w+)",  # 3 groups
                revision_format="step{}-tokens{}B",
            )

    def test_model_config_no_regex(self):
        """Test model configuration without regex (no checkpoint parsing)."""
        config = ModelConfig(hf_name="test/model")

        assert config.checkpoints == []
        assert not hasattr(config, "all_branches")

    def test_model_config_branch_regex_compilation(self):
        """Test that branch regex is properly compiled."""
        with patch("huggingface_hub.list_repo_refs") as mock_list_refs:
            mock_list_refs.return_value = mock_huggingface_repo_refs([])

            config = ModelConfig(
                hf_name="test/model",
                branch_regex=r"step(\d+)",
                revision_format="step{}",
            )

            assert isinstance(config.branch_regex, re.Pattern)

    @patch("huggingface_hub.list_repo_refs")
    def test_model_config_stores_all_branches(self, mock_list_refs):
        """Test that model config stores all branches."""
        branches = ["step1000", "main", "dev"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model", branch_regex=r"step(\d+)", revision_format="step{}"
        )

        assert hasattr(config, "all_branches")
        assert set(config.all_branches) == set(branches)


class TestModelsRegistry:
    """Test the MODELS registry."""

    def test_models_registry_structure(self):
        """Test that MODELS registry is properly structured."""
        assert isinstance(MODELS, dict)
        assert len(MODELS) > 0

        # Check that all entries are ModelConfig instances
        for name, config in MODELS.items():
            assert isinstance(name, str)
            assert isinstance(config, ModelConfig)

    def test_olmoe_model_config(self):
        """Test OLMoE model configuration."""
        assert "olmoe" in MODELS
        olmoe = MODELS["olmoe"]

        assert olmoe.hf_name == "allenai/OLMoE-1B-7B-0924"
        assert olmoe.branch_regex is not None
        assert olmoe.revision_format == "step{}-tokens{}B"
        assert olmoe.tokenizer_has_padding_token is True  # default

    def test_phimoe_model_config(self):
        """Test PhiMoE model configuration."""
        assert "phimoe" in MODELS
        phimoe = MODELS["phimoe"]

        assert phimoe.hf_name == "microsoft/Phi-3.5-MoE-instruct"
        assert phimoe.branch_regex is None
        assert phimoe.revision_format is None

    def test_q3_30b_model_config(self):
        """Test Qwen3-30B model configuration."""
        assert "q3_30b" in MODELS
        q3_30b = MODELS["q3_30b"]

        assert q3_30b.hf_name == "Qwen/Qwen3-30B-A3B"
        assert q3_30b.branch_regex is None
        assert q3_30b.revision_format is None

    def test_all_models_have_valid_hf_names(self):
        """Test that all models have valid HuggingFace names."""
        for config in MODELS.values():
            assert isinstance(config.hf_name, str)
            assert len(config.hf_name) > 0
            assert "/" in config.hf_name  # Should be in format "org/model"


class TestModelConfigIntegration:
    """Integration tests for model configuration."""

    @patch("huggingface_hub.list_repo_refs")
    def test_checkpoint_model_config_reference(self, mock_list_refs):
        """Test that checkpoints maintain reference to their model config."""
        branches = ["step1000-tokens1B"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
        )

        checkpoint = config.checkpoints[0]
        assert checkpoint.model_config is config
        assert checkpoint.model_config.hf_name == "test/model"

    @patch("huggingface_hub.list_repo_refs")
    def test_checkpoint_sorting_stability(self, mock_list_refs):
        """Test that checkpoint sorting is stable and correct."""
        branches = [
            "step3000-tokens3B",
            "step1000-tokens1B",
            "step2000-tokens2B",
            "step1500-tokens1B",  # Same tokens, different step
        ]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
        )

        steps = [c.step for c in config.checkpoints]
        assert steps == sorted(steps), "Checkpoints should be sorted by step"
        assert steps == [1000, 1500, 2000, 3000]

    def test_model_config_defaults(self):
        """Test model configuration default values."""
        config = ModelConfig(hf_name="test/model")

        assert config.tokenizer_has_padding_token is True
        assert config.checkpoints == []
        assert config.branch_regex is None
        assert config.revision_format is None

    @patch("huggingface_hub.list_repo_refs")
    def test_regex_pattern_validation(self, mock_list_refs):
        """Test that regex patterns work correctly with various branch names."""
        branches = [
            "step1000-tokens1B",
            "step2000-tokens10B",
            "step3000-tokens100B",
            "checkpoint-1000",  # Should not match
            "step-invalid",  # Should not match
        ]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
        )

        # Should only match the first 3 branches
        assert len(config.checkpoints) == 3

        # Check that the parsing worked correctly
        tokens = [c.num_tokens for c in config.checkpoints]
        assert set(tokens) == {1, 10, 100}


class TestModelConfigErrorHandling:
    """Test error handling in model configuration."""

    @patch("huggingface_hub.list_repo_refs")
    def test_huggingface_api_error_handling(self, mock_list_refs):
        """Test handling of HuggingFace API errors."""
        # Simulate API error
        mock_list_refs.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            ModelConfig(
                hf_name="test/model",
                branch_regex=r"step(\d+)",
                revision_format="step{}",
            )

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex patterns."""
        with patch("huggingface_hub.list_repo_refs") as mock_list_refs:
            mock_list_refs.return_value = mock_huggingface_repo_refs([])

            with pytest.raises(re.error):
                ModelConfig(
                    hf_name="test/model",
                    branch_regex="[invalid regex",  # Invalid regex
                    revision_format="step{}",
                )

    @patch("huggingface_hub.list_repo_refs")
    def test_non_numeric_step_handling(self, mock_list_refs):
        """Test handling of non-numeric step values."""
        branches = ["step-abc-tokens1B"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        with pytest.raises(ValueError):
            ModelConfig(
                hf_name="test/model",
                branch_regex=r"step-(\w+)-tokens(\d+)B",
                revision_format="step-{}-tokens{}B",
            )

    @patch("huggingface_hub.list_repo_refs")
    def test_non_numeric_tokens_handling(self, mock_list_refs):
        """Test handling of non-numeric token values."""
        branches = ["step1000-tokens-abc-B"]
        mock_list_refs.return_value = mock_huggingface_repo_refs(branches)

        with pytest.raises(ValueError):
            ModelConfig(
                hf_name="test/model",
                branch_regex=r"step(\d+)-tokens-(\w+)-B",
                revision_format="step{}-tokens-{}-B",
            )
