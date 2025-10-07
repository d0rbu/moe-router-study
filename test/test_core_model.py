"""Tests for core.model module."""

import re
from unittest.mock import MagicMock, patch

import pytest

from core.model import (
    LATEST_REVISION,
    MODELS,
    Checkpoint,
    ModelConfig,
    get_model_config,
)
from test.utils import create_mock_hf_refs


class TestCheckpoint:
    """Test Checkpoint dataclass."""
    
    def test_checkpoint_creation(self):
        """Test basic checkpoint creation."""
        model_config = ModelConfig(hf_name="test/model")
        checkpoint = Checkpoint(
            step=1000,
            num_tokens=5000,
            model_config=model_config
        )
        
        assert checkpoint.step == 1000
        assert checkpoint.num_tokens == 5000
        assert checkpoint.model_config is model_config
        assert checkpoint.revision is None
    
    def test_checkpoint_with_revision(self):
        """Test checkpoint with explicit revision."""
        model_config = ModelConfig(hf_name="test/model")
        checkpoint = Checkpoint(
            step=1000,
            num_tokens=5000,
            model_config=model_config,
            revision="custom-branch"
        )
        
        assert checkpoint.revision == "custom-branch"
    
    def test_checkpoint_str_with_revision(self):
        """Test string representation with explicit revision."""
        model_config = ModelConfig(hf_name="test/model")
        checkpoint = Checkpoint(
            step=1000,
            num_tokens=5000,
            model_config=model_config,
            revision="main"
        )
        
        assert str(checkpoint) == "main"
    
    def test_checkpoint_str_with_format(self):
        """Test string representation with revision format."""
        model_config = ModelConfig(
            hf_name="test/model",
            revision_format="step{}-tokens{}B"
        )
        checkpoint = Checkpoint(
            step=1000,
            num_tokens=5000,
            model_config=model_config
        )
        
        assert str(checkpoint) == "step1000-tokens5000B"
    
    def test_checkpoint_str_no_format_error(self):
        """Test string representation fails without revision format."""
        model_config = ModelConfig(hf_name="test/model")
        checkpoint = Checkpoint(
            step=1000,
            num_tokens=5000,
            model_config=model_config
        )
        
        with pytest.raises(ValueError, match="revision_format is required"):
            str(checkpoint)
    
    def test_checkpoint_str_none_values(self):
        """Test string representation with None values."""
        model_config = ModelConfig(
            hf_name="test/model",
            revision_format="step{}-tokens{}B"
        )
        checkpoint = Checkpoint(
            step=None,
            num_tokens=None,
            model_config=model_config
        )
        
        assert str(checkpoint) == "stepNone-tokensNoneB"


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test basic model config creation."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        
        assert config.hf_name == "test/model"
        assert config.branch_regex is None
        assert config.revision_format is None
        assert config.total_steps is None
        assert config.total_tokens is None
        assert config.tokenizer_has_padding_token is True
        assert config.checkpoints == []
        assert config.eager_fetch is False
    
    def test_model_config_with_parameters(self):
        """Test model config with all parameters."""
        regex_pattern = re.compile(r"step(\d+)")
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=regex_pattern,
            revision_format="step{}",
            total_steps=10000,
            total_tokens=50000,
            tokenizer_has_padding_token=False,
            eager_fetch=False
        )
        
        assert config.hf_name == "test/model"
        assert config.branch_regex == regex_pattern
        assert config.revision_format == "step{}"
        assert config.total_steps == 10000
        assert config.total_tokens == 50000
        assert config.tokenizer_has_padding_token is False
        assert config.eager_fetch is False
    
    def test_latest_checkpoint_property(self):
        """Test latest_checkpoint property."""
        config = ModelConfig(
            hf_name="test/model",
            total_steps=1000,
            total_tokens=5000,
            eager_fetch=False
        )
        
        latest = config.latest_checkpoint
        assert isinstance(latest, Checkpoint)
        assert latest.step == 1000
        assert latest.num_tokens == 5000
        assert latest.model_config is config
        assert latest.revision == LATEST_REVISION
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_no_regex(self, mock_list_refs):
        """Test fetch_checkpoints when no regex is provided."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        
        result = config.fetch_checkpoints()
        
        # Should return only the latest checkpoint
        assert len(result) == 1
        assert result[0].revision == LATEST_REVISION
        # Should not call HuggingFace API
        mock_list_refs.assert_not_called()
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_with_regex(self, mock_list_refs):
        """Test fetch_checkpoints with regex pattern."""
        # Mock HuggingFace response
        mock_refs = create_mock_hf_refs([
            "step1000-tokens5000B",
            "step2000-tokens10000B", 
            "main",
            "dev",
            "step500-tokens2500B"
        ])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
            eager_fetch=False
        )
        
        result = config.fetch_checkpoints()
        
        # Should have parsed checkpoints + latest
        assert len(result) == 4  # 3 parsed + 1 latest
        
        # Check that checkpoints are sorted
        steps = [cp.step for cp in result[:-1]]  # Exclude latest
        assert steps == sorted(steps)
        
        # Check specific checkpoints
        checkpoint_steps = {cp.step for cp in result if cp.revision != LATEST_REVISION}
        assert checkpoint_steps == {500, 1000, 2000}
        
        mock_list_refs.assert_called_once_with("test/model")
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_string_regex(self, mock_list_refs):
        """Test fetch_checkpoints with string regex pattern."""
        mock_refs = create_mock_hf_refs(["step1000", "step2000", "main"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)",  # String instead of compiled regex
            revision_format="step{}",
            eager_fetch=False
        )
        
        result = config.fetch_checkpoints()
        
        # Should work with string regex
        assert len(result) == 3  # 2 parsed + 1 latest
        checkpoint_steps = {cp.step for cp in result if cp.revision != LATEST_REVISION}
        assert checkpoint_steps == {1000, 2000}
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_no_matches(self, mock_list_refs):
        """Test fetch_checkpoints when no branches match regex."""
        mock_refs = create_mock_hf_refs(["main", "dev", "feature-branch"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)",
            revision_format="step{}",
            eager_fetch=False
        )
        
        result = config.fetch_checkpoints()
        
        # Should return only latest checkpoint
        assert len(result) == 1
        assert result[0].revision == LATEST_REVISION
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_validation_error(self, mock_list_refs):
        """Test fetch_checkpoints with validation errors."""
        mock_refs = create_mock_hf_refs(["step10000-tokens50000B"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
            total_steps=5000,  # Less than checkpoint step
            total_tokens=25000,  # Less than checkpoint tokens
            eager_fetch=False
        )
        
        with pytest.raises(AssertionError, match="total_steps.*is less than"):
            config.fetch_checkpoints()
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_single_group_regex(self, mock_list_refs):
        """Test fetch_checkpoints with single group regex."""
        mock_refs = create_mock_hf_refs(["step1000", "step2000"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)",
            revision_format="step{}",
            eager_fetch=False
        )
        
        result = config.fetch_checkpoints()
        
        # Should handle single group (step only, no tokens)
        assert len(result) == 3  # 2 parsed + 1 latest
        for cp in result:
            if cp.revision != LATEST_REVISION:
                assert cp.num_tokens is None
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_invalid_groups(self, mock_list_refs):
        """Test fetch_checkpoints with invalid number of groups."""
        mock_refs = create_mock_hf_refs(["step1000-tokens5000-extra"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)-(\w+)",  # 3 groups
            revision_format="step{}-tokens{}-{}",
            eager_fetch=False
        )
        
        with pytest.raises(ValueError, match="Unexpected number of groups"):
            config.fetch_checkpoints()
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_fetch_checkpoints_duplicate_latest(self, mock_list_refs):
        """Test that duplicate latest checkpoint is not added."""
        mock_refs = create_mock_hf_refs(["step1000-tokens5000B"])
        mock_list_refs.return_value = mock_refs
        
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
            total_steps=1000,  # Same as checkpoint
            total_tokens=5000,  # Same as checkpoint
            eager_fetch=False
        )
        
        result = config.fetch_checkpoints()
        
        # Should not duplicate the latest checkpoint
        assert len(result) == 1
        assert result[0].step == 1000
        assert result[0].num_tokens == 5000
    
    def test_post_init_eager_fetch(self):
        """Test that __post_init__ calls fetch_checkpoints when eager_fetch=True."""
        with patch.object(ModelConfig, 'fetch_checkpoints') as mock_fetch:
            mock_fetch.return_value = []
            
            config = ModelConfig(hf_name="test/model", eager_fetch=True)
            
            mock_fetch.assert_called_once()
    
    def test_post_init_no_eager_fetch(self):
        """Test that __post_init__ doesn't call fetch_checkpoints when eager_fetch=False."""
        with patch.object(ModelConfig, 'fetch_checkpoints') as mock_fetch:
            config = ModelConfig(hf_name="test/model", eager_fetch=False)
            
            mock_fetch.assert_not_called()
    
    def test_get_checkpoint_no_params(self):
        """Test get_checkpoint with no parameters."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [
            Checkpoint(1000, 5000, config),
            Checkpoint(2000, 10000, config)
        ]
        
        with patch('core.model.logger') as mock_logger:
            result = config.get_checkpoint()
            
            # Should return last checkpoint and log warning
            assert result.step == 2000
            mock_logger.warning.assert_called_once()
    
    def test_get_checkpoint_by_step(self):
        """Test get_checkpoint by step."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [
            Checkpoint(1000, 5000, config),
            Checkpoint(2000, 10000, config),
            Checkpoint(2000, 15000, config)  # Same step, different tokens
        ]
        
        result = config.get_checkpoint(step=2000)
        
        # Should return the last checkpoint with matching step
        assert result.step == 2000
        assert result.num_tokens == 15000
    
    def test_get_checkpoint_by_tokens(self):
        """Test get_checkpoint by num_tokens."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [
            Checkpoint(1000, 5000, config),
            Checkpoint(2000, 10000, config)
        ]
        
        result = config.get_checkpoint(num_tokens=5000)
        
        assert result.step == 1000
        assert result.num_tokens == 5000
    
    def test_get_checkpoint_by_both(self):
        """Test get_checkpoint by both step and num_tokens."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [
            Checkpoint(1000, 5000, config),
            Checkpoint(1000, 10000, config),
            Checkpoint(2000, 5000, config)
        ]
        
        result = config.get_checkpoint(step=1000, num_tokens=5000)
        
        assert result.step == 1000
        assert result.num_tokens == 5000
    
    def test_get_checkpoint_not_found(self):
        """Test get_checkpoint when no matching checkpoint exists."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [Checkpoint(1000, 5000, config)]
        
        result = config.get_checkpoint(step=9999)
        
        assert result is None
    
    def test_get_checkpoint_lazy_fetch(self):
        """Test get_checkpoint triggers fetch when checkpoints is empty."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        assert config.checkpoints == []
        
        with patch.object(config, 'fetch_checkpoints') as mock_fetch:
            mock_fetch.return_value = [Checkpoint(1000, 5000, config)]
            
            result = config.get_checkpoint(step=1000)
            
            mock_fetch.assert_called_once()
            assert result.step == 1000
    
    def test_get_checkpoint_strict_success(self):
        """Test get_checkpoint_strict when checkpoint exists."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [Checkpoint(1000, 5000, config)]
        
        result = config.get_checkpoint_strict(step=1000)
        
        assert result.step == 1000
    
    def test_get_checkpoint_strict_failure(self):
        """Test get_checkpoint_strict when checkpoint doesn't exist."""
        config = ModelConfig(hf_name="test/model", eager_fetch=False)
        config.checkpoints = [Checkpoint(1000, 5000, config)]
        
        with pytest.raises(ValueError, match="Checkpoint for step 9999.*not found"):
            config.get_checkpoint_strict(step=9999)


class TestModelsConstant:
    """Test MODELS constant."""
    
    def test_models_structure(self):
        """Test that MODELS has expected structure."""
        assert isinstance(MODELS, dict)
        assert len(MODELS) > 0
        
        for name, config in MODELS.items():
            assert isinstance(name, str)
            assert isinstance(config, ModelConfig)
            assert config.hf_name is not None
    
    def test_models_content(self):
        """Test that expected models are present."""
        expected_models = ["olmoe", "phimoe", "q3_30b", "gpt", "olmoe-i"]
        
        for model_name in expected_models:
            assert model_name in MODELS
    
    def test_olmoe_model_config(self):
        """Test specific olmoe model configuration."""
        olmoe = MODELS["olmoe"]
        
        assert olmoe.hf_name == "allenai/OLMoE-1B-7B-0924"
        assert olmoe.branch_regex is not None
        assert olmoe.revision_format == "step{}-tokens{}B"
        assert olmoe.eager_fetch is False
        assert olmoe.total_steps == 1_223_842
        assert olmoe.total_tokens == 5133


class TestGetModelConfig:
    """Test get_model_config function."""
    
    def test_get_model_config_valid(self):
        """Test get_model_config with valid model name."""
        config = get_model_config("olmoe")
        
        assert isinstance(config, ModelConfig)
        assert config.hf_name == "allenai/OLMoE-1B-7B-0924"
    
    def test_get_model_config_invalid(self):
        """Test get_model_config with invalid model name."""
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            get_model_config("nonexistent")
    
    def test_get_model_config_all_models(self):
        """Test get_model_config with all defined models."""
        for model_name in MODELS.keys():
            config = get_model_config(model_name)
            assert isinstance(config, ModelConfig)
            assert config is MODELS[model_name]


class TestLatestRevisionConstant:
    """Test LATEST_REVISION constant."""
    
    def test_latest_revision_value(self):
        """Test that LATEST_REVISION has expected value."""
        assert LATEST_REVISION == "main"
        assert isinstance(LATEST_REVISION, str)


class TestModelIntegration:
    """Integration tests for model functionality."""
    
    @patch('core.model.huggingface_hub.list_repo_refs')
    def test_full_workflow(self, mock_list_refs):
        """Test a complete workflow with model config."""
        # Mock HuggingFace response
        mock_refs = create_mock_hf_refs([
            "step1000-tokens5000B",
            "step2000-tokens10000B",
            "main"
        ])
        mock_list_refs.return_value = mock_refs
        
        # Create config and fetch checkpoints
        config = ModelConfig(
            hf_name="test/model",
            branch_regex=r"step(\d+)-tokens(\d+)B",
            revision_format="step{}-tokens{}B",
            total_steps=3000,
            total_tokens=15000,
            eager_fetch=False
        )
        
        checkpoints = config.fetch_checkpoints()
        
        # Verify checkpoints
        assert len(checkpoints) == 3  # 2 parsed + 1 latest
        
        # Test checkpoint retrieval
        cp1000 = config.get_checkpoint(step=1000)
        assert cp1000 is not None
        assert str(cp1000) == "step1000-tokens5000B"
        
        # Test latest checkpoint
        latest = config.get_checkpoint_strict(step=3000, num_tokens=15000)
        assert latest.revision == LATEST_REVISION
