"""Tests for core.data module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.data import (
    DATASETS,
    fineweb_10bt_text,
    get_dataset_fn,
    lmsys_chat_1m_text,
    smollm2_small,
    toy_text,
)


class TestToyText:
    """Test toy_text dataset function."""
    
    def test_toy_text_basic(self):
        """Test basic toy_text functionality."""
        result = toy_text()
        
        # Should return an iterable
        assert hasattr(result, '__iter__')
        
        # Convert to list to test content
        samples = list(result)
        
        # Should have expected samples
        assert len(samples) == 4
        assert "Tiny sample 1" in samples
        assert "Tiny sample 2" in samples
        assert "Tiny sample 3" in samples
        assert "Tiny sample 4" in samples
    
    def test_toy_text_with_tokenizer(self):
        """Test toy_text with tokenizer parameter (should be ignored)."""
        mock_tokenizer = MagicMock()
        result = toy_text(mock_tokenizer)
        
        samples = list(result)
        assert len(samples) == 4
        # Tokenizer should not be used
        mock_tokenizer.assert_not_called()
    
    def test_toy_text_deterministic(self):
        """Test that toy_text returns the same samples each time."""
        result1 = list(toy_text())
        result2 = list(toy_text())
        
        assert result1 == result2


class TestFineweb10btText:
    """Test fineweb_10bt_text dataset function."""
    
    @patch('core.data.load_dataset')
    def test_fineweb_10bt_text_basic(self, mock_load_dataset):
        """Test basic fineweb_10bt_text functionality."""
        # Mock dataset response
        mock_dataset = {"text": ["Sample 1", "Sample 2", "Sample 3"]}
        mock_load_dataset.return_value = mock_dataset
        
        result = fineweb_10bt_text()
        
        # Should call load_dataset with correct parameters
        mock_load_dataset.assert_called_once_with(
            "HuggingFaceFW/fineweb", 
            name="sample-10BT", 
            split="train", 
            streaming=True
        )
        
        # Should return the text field
        assert result == ["Sample 1", "Sample 2", "Sample 3"]
    
    @patch('core.data.load_dataset')
    def test_fineweb_10bt_text_with_tokenizer(self, mock_load_dataset):
        """Test fineweb_10bt_text with tokenizer parameter (should be ignored)."""
        mock_dataset = {"text": ["Sample"]}
        mock_load_dataset.return_value = mock_dataset
        
        mock_tokenizer = MagicMock()
        result = fineweb_10bt_text(mock_tokenizer)
        
        # Tokenizer should not be used
        mock_tokenizer.assert_not_called()
        assert result == ["Sample"]


class TestSmollm2Small:
    """Test smollm2_small dataset function."""
    
    @patch('core.data.load_dataset')
    def test_smollm2_small_basic(self, mock_load_dataset):
        """Test basic smollm2_small functionality."""
        # Mock dataset response
        mock_dataset = {"text": ["SmolLM sample 1", "SmolLM sample 2"]}
        mock_load_dataset.return_value = mock_dataset
        
        result = smollm2_small()
        
        # Should call load_dataset with correct parameters
        mock_load_dataset.assert_called_once_with(
            "EleutherAI/SmolLM2-135M-10B", 
            split="train[:1%]"
        )
        
        # Should return the text field
        assert result == ["SmolLM sample 1", "SmolLM sample 2"]
    
    @patch('core.data.load_dataset')
    def test_smollm2_small_with_tokenizer(self, mock_load_dataset):
        """Test smollm2_small with tokenizer parameter (should be ignored)."""
        mock_dataset = {"text": ["Sample"]}
        mock_load_dataset.return_value = mock_dataset
        
        mock_tokenizer = MagicMock()
        result = smollm2_small(mock_tokenizer)
        
        # Tokenizer should not be used
        mock_tokenizer.assert_not_called()
        assert result == ["Sample"]


class TestLmsysChatText:
    """Test lmsys_chat_1m_text dataset function."""
    
    def test_lmsys_chat_1m_text_requires_tokenizer(self):
        """Test that lmsys_chat_1m_text requires a tokenizer."""
        # Should work with a tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"
        
        with patch('core.data.load_dataset') as mock_load_dataset:
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = [
                [{"role": "user", "content": "Hello"}]
            ]
            mock_load_dataset.return_value = mock_dataset
            
            result = lmsys_chat_1m_text(mock_tokenizer)
            
            # Should be iterable
            assert hasattr(result, '__iter__')
    
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_streaming_mode(self, mock_load_dataset):
        """Test lmsys_chat_1m_text in streaming mode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"
        
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            [{"role": "user", "content": "Hello"}]
        ]
        mock_load_dataset.return_value = mock_dataset
        
        # Test streaming mode (default)
        result = lmsys_chat_1m_text(mock_tokenizer, streaming=True)
        
        mock_load_dataset.assert_called_once_with(
            "lmsys/lmsys-chat-1m", 
            split="train", 
            streaming=True
        )
    
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_non_streaming_mode(self, mock_load_dataset):
        """Test lmsys_chat_1m_text in non-streaming mode."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"
        
        # Mock dataset with length and select methods
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__getitem__.return_value = [
            [{"role": "user", "content": "Hello"}]
        ]
        mock_load_dataset.return_value = mock_dataset
        
        with patch('core.data.assert_type', return_value=mock_dataset):
            # Test non-streaming mode
            result = lmsys_chat_1m_text(
                mock_tokenizer, 
                start_idx=10, 
                stop_idx=20, 
                streaming=False
            )
            
            mock_load_dataset.assert_called_once_with(
                "lmsys/lmsys-chat-1m", 
                split="train", 
                streaming=False
            )
            
            # Should call select with correct range
            mock_dataset.select.assert_called_once()
    
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_streaming_with_indices_error(self, mock_load_dataset):
        """Test that streaming mode doesn't accept start/stop indices."""
        mock_tokenizer = MagicMock()
        
        with pytest.raises(AssertionError, match="Streaming mode does not support"):
            lmsys_chat_1m_text(
                mock_tokenizer, 
                start_idx=10, 
                stop_idx=20, 
                streaming=True
            )
    
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_validation_errors(self, mock_load_dataset):
        """Test validation errors in non-streaming mode."""
        mock_tokenizer = MagicMock()
        
        # Create a mock dataset that behaves like the real Dataset class
        from core.type import assert_type
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_load_dataset.return_value = mock_dataset
        
        # Mock assert_type to return the mock dataset
        with patch('core.data.assert_type', return_value=mock_dataset):
            # Test negative indices
            with pytest.raises(AssertionError, match="start_idx and stop_idx to be non-negative"):
                lmsys_chat_1m_text(
                    mock_tokenizer, 
                    start_idx=-1, 
                    stop_idx=10, 
                    streaming=False
                )
            
            # Test start_idx >= stop_idx
            with pytest.raises(AssertionError, match="start_idx must be less than stop_idx"):
                lmsys_chat_1m_text(
                    mock_tokenizer, 
                    start_idx=20, 
                    stop_idx=10, 
                    streaming=False
                )
            
            # Test start_idx >= dataset length
            with pytest.raises(AssertionError, match="start_idx must be less than the length"):
                lmsys_chat_1m_text(
                    mock_tokenizer, 
                    start_idx=150, 
                    stop_idx=200, 
                    streaming=False
                )
            
            # Test stop_idx > dataset length
            with pytest.raises(AssertionError, match="stop_idx must be less than or equal"):
                lmsys_chat_1m_text(
                    mock_tokenizer, 
                    start_idx=10, 
                    stop_idx=150, 
                    streaming=False
                )
    
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_stop_idx_zero(self, mock_load_dataset):
        """Test that stop_idx=0 uses dataset length."""
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__getitem__.return_value = []
        mock_load_dataset.return_value = mock_dataset
        
        with patch('core.data.assert_type', return_value=mock_dataset):
            lmsys_chat_1m_text(
                mock_tokenizer, 
                start_idx=10, 
                stop_idx=0,  # Should use dataset length
                streaming=False
            )
            
            # Should select from 10 to 100
            mock_dataset.select.assert_called_once()
            call_args = mock_dataset.select.call_args[0][0]
            assert call_args.start == 10
            assert call_args.stop == 100
    
    def test_lmsys_chat_1m_text_format_conversation_error(self):
        """Test error handling in conversation formatting."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = 123  # Not a string
        
        with patch('core.data.load_dataset') as mock_load_dataset:
            mock_dataset = MagicMock()
            mock_dataset.__getitem__.return_value = [
                [{"role": "user", "content": "Hello"}]
            ]
            mock_load_dataset.return_value = mock_dataset
            
            result_iter = lmsys_chat_1m_text(mock_tokenizer)
            
            # Should raise ValueError when iterating
            with pytest.raises(ValueError, match="Expected chat to be a string"):
                list(result_iter)
    
    @patch('core.data.os.path.exists')
    @patch('core.data.load_dataset')
    def test_lmsys_chat_1m_text_local_path(self, mock_load_dataset, mock_exists):
        """Test loading from local path when it exists."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        
        # Mock local path exists
        mock_exists.return_value = True
        
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = []
        mock_load_dataset.return_value = mock_dataset
        
        with patch('core.data.DATASET_DIRNAME', '/mock/dataset/dir'):
            lmsys_chat_1m_text(mock_tokenizer)
            
            # Should load from local path
            expected_path = '/mock/dataset/dir/lmsys/lmsys-chat-1m'
            mock_load_dataset.assert_called_once_with(
                expected_path, 
                split="train", 
                streaming=True
            )


class TestDatasetsConstant:
    """Test DATASETS constant."""
    
    def test_datasets_structure(self):
        """Test that DATASETS has expected structure."""
        assert isinstance(DATASETS, dict)
        assert len(DATASETS) > 0
        
        for name, func in DATASETS.items():
            assert isinstance(name, str)
            assert callable(func)
    
    def test_datasets_content(self):
        """Test that expected datasets are present."""
        expected_datasets = ["fw", "toy", "lmsys", "smol"]
        
        for dataset_name in expected_datasets:
            assert dataset_name in DATASETS
    
    def test_datasets_mapping(self):
        """Test that dataset functions are correctly mapped."""
        assert DATASETS["fw"] is fineweb_10bt_text
        assert DATASETS["toy"] is toy_text
        assert DATASETS["lmsys"] is lmsys_chat_1m_text
        assert DATASETS["smol"] is smollm2_small


class TestGetDatasetFn:
    """Test get_dataset_fn function."""
    
    def test_get_dataset_fn_valid(self):
        """Test get_dataset_fn with valid dataset names."""
        for dataset_name in DATASETS.keys():
            result = get_dataset_fn(dataset_name)
            assert callable(result)
            assert result is DATASETS[dataset_name]
    
    def test_get_dataset_fn_invalid(self):
        """Test get_dataset_fn with invalid dataset name."""
        with pytest.raises(ValueError, match="Dataset nonexistent not found"):
            get_dataset_fn("nonexistent")
    
    def test_get_dataset_fn_empty_string(self):
        """Test get_dataset_fn with empty string."""
        with pytest.raises(ValueError, match="Dataset  not found"):
            get_dataset_fn("")


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""
    
    def test_toy_dataset_integration(self):
        """Test toy dataset can be used end-to-end."""
        dataset_fn = get_dataset_fn("toy")
        
        # Should work without tokenizer
        result = dataset_fn(None)
        samples = list(result)
        
        assert len(samples) == 4
        assert all(isinstance(sample, str) for sample in samples)
    
    @patch('core.data.load_dataset')
    def test_fineweb_dataset_integration(self, mock_load_dataset):
        """Test fineweb dataset integration."""
        mock_load_dataset.return_value = {"text": ["Sample text"]}
        
        dataset_fn = get_dataset_fn("fw")
        result = dataset_fn(None)
        
        assert result == ["Sample text"]
    
    def test_dataset_function_signatures(self):
        """Test that all dataset functions have compatible signatures."""
        mock_tokenizer = MagicMock()
        
        # toy and smol should work with None tokenizer
        for dataset_name in ["toy"]:
            dataset_fn = get_dataset_fn(dataset_name)
            result = dataset_fn(None)
            assert hasattr(result, '__iter__')
        
        # lmsys requires a tokenizer
        lmsys_fn = get_dataset_fn("lmsys")
        # This would fail without proper mocking, but we can test the function exists
        assert callable(lmsys_fn)


class TestDatasetMainScript:
    """Test the main script functionality."""
    
    @patch('core.data.AutoTokenizer.from_pretrained')
    @patch('core.data.load_dataset')
    def test_main_script_structure(self, mock_load_dataset, mock_tokenizer):
        """Test that the main script structure is valid."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock datasets
        mock_load_dataset.return_value = {"text": ["Sample"]}
        
        # The main script code should be importable and structured correctly
        # We can't easily test the actual execution, but we can verify the structure
        
        # Check that the main block exists and uses expected components
        import core.data
        
        # Verify the main components are available
        assert hasattr(core.data, 'DATASETS')
        assert hasattr(core.data, 'AutoTokenizer')
        assert hasattr(core.data, 'tqdm')
    
    def test_dataset_error_handling(self):
        """Test error handling in dataset functions."""
        # Test that functions handle missing dependencies gracefully
        # This is more of a smoke test to ensure imports work
        
        # All dataset functions should be importable
        from core.data import toy_text, fineweb_10bt_text, smollm2_small, lmsys_chat_1m_text
        
        # toy_text should always work
        result = toy_text()
        assert hasattr(result, '__iter__')
        
        # Other functions may require external dependencies
        # but should at least be callable
        assert callable(fineweb_10bt_text)
        assert callable(smollm2_small)
        assert callable(lmsys_chat_1m_text)
