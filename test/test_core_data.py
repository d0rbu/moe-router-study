"""Tests for core.data module."""

from unittest.mock import MagicMock, patch

import pytest

from core.data import DATASETS, fineweb_10bt_text


class TestFineweb10btText:
    """Test fineweb_10bt_text function."""
    
    @patch('datasets.load_dataset')
    def test_fineweb_10bt_text_basic(self, mock_load_dataset):
        """Test basic functionality of fineweb_10bt_text."""
        # Mock the dataset
        mock_dataset = MagicMock()
        mock_text_column = ["Sample text 1", "Sample text 2", "Sample text 3"]
        mock_dataset.__getitem__.return_value = mock_text_column
        mock_load_dataset.return_value = mock_dataset
        
        result = fineweb_10bt_text()
        
        # Verify dataset was loaded with correct parameters
        mock_load_dataset.assert_called_once_with(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        
        # Verify we get the text column
        assert result == mock_text_column
    
    @patch('datasets.load_dataset')
    def test_fineweb_10bt_text_streaming_enabled(self, mock_load_dataset):
        """Test that streaming is enabled for fineweb dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        fineweb_10bt_text()
        
        # Verify streaming=True was passed
        call_args = mock_load_dataset.call_args
        assert call_args[1]['streaming'] is True
    
    @patch('datasets.load_dataset')
    def test_fineweb_10bt_text_correct_split(self, mock_load_dataset):
        """Test that correct split is used for fineweb dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        fineweb_10bt_text()
        
        # Verify split="train" was passed
        call_args = mock_load_dataset.call_args
        assert call_args[1]['split'] == "train"
    
    @patch('datasets.load_dataset')
    def test_fineweb_10bt_text_error_handling(self, mock_load_dataset):
        """Test error handling in fineweb_10bt_text."""
        # Simulate dataset loading error
        mock_load_dataset.side_effect = Exception("Dataset loading failed")
        
        with pytest.raises(Exception, match="Dataset loading failed"):
            fineweb_10bt_text()


class TestDatasetsRegistry:
    """Test the DATASETS registry."""
    
    def test_datasets_registry_structure(self):
        """Test that DATASETS registry is properly structured."""
        assert isinstance(DATASETS, dict)
        assert len(DATASETS) > 0
        
        # Check that all entries are callable
        for name, func in DATASETS.items():
            assert isinstance(name, str)
            assert callable(func)
    
    def test_fineweb_in_registry(self):
        """Test that fineweb dataset is in registry."""
        assert "fw" in DATASETS
        assert DATASETS["fw"] == fineweb_10bt_text
    
    def test_registry_functions_callable(self):
        """Test that all registry functions are callable."""
        for name, func in DATASETS.items():
            assert callable(func), f"Dataset function {name} should be callable"
    
    @patch('datasets.load_dataset')
    def test_registry_functions_return_iterable(self, mock_load_dataset):
        """Test that registry functions return iterable objects."""
        # Mock dataset with iterable text column
        mock_dataset = MagicMock()
        mock_text_column = ["text1", "text2", "text3"]
        mock_dataset.__getitem__.return_value = mock_text_column
        mock_load_dataset.return_value = mock_dataset
        
        for name, func in DATASETS.items():
            result = func()
            # Should be iterable (has __iter__ or __getitem__)
            assert hasattr(result, '__iter__') or hasattr(result, '__getitem__'), \
                f"Dataset function {name} should return iterable"


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""
    
    @patch('datasets.load_dataset')
    def test_dataset_iteration_pattern(self, mock_load_dataset):
        """Test the expected dataset iteration pattern."""
        # Create a mock iterable dataset
        mock_text_data = ["Sample 1", "Sample 2", "Sample 3"]
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = iter(mock_text_data)
        mock_load_dataset.return_value = mock_dataset
        
        dataset_func = DATASETS["fw"]
        text_column = dataset_func()
        
        # Should be able to iterate over the text column
        collected_texts = list(text_column)
        assert collected_texts == mock_text_data
    
    @patch('datasets.load_dataset')
    def test_multiple_dataset_calls_consistency(self, mock_load_dataset):
        """Test that multiple calls to dataset functions are consistent."""
        mock_dataset = MagicMock()
        mock_text_column = ["consistent", "data"]
        mock_dataset.__getitem__.return_value = mock_text_column
        mock_load_dataset.return_value = mock_dataset
        
        dataset_func = DATASETS["fw"]
        
        # Call multiple times
        result1 = dataset_func()
        result2 = dataset_func()
        
        # Should get same result (though this depends on implementation)
        assert result1 == result2
        
        # Should have called load_dataset multiple times
        assert mock_load_dataset.call_count == 2
    
    @patch('datasets.load_dataset')
    def test_dataset_error_propagation(self, mock_load_dataset):
        """Test that dataset errors are properly propagated."""
        mock_load_dataset.side_effect = ConnectionError("Network error")
        
        dataset_func = DATASETS["fw"]
        
        with pytest.raises(ConnectionError, match="Network error"):
            dataset_func()


class TestDatasetMainExecution:
    """Test the main execution block of data.py."""
    
    @patch('datasets.load_dataset')
    @patch('tqdm.tqdm')
    def test_main_execution_mock(self, mock_tqdm, mock_load_dataset):
        """Test the main execution block with mocking."""
        # Mock dataset
        mock_text_data = ["text1", "text2", "text3"]
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = iter(mock_text_data)
        mock_load_dataset.return_value = mock_dataset
        
        # Mock tqdm to return the input unchanged
        mock_tqdm.side_effect = lambda x, **kwargs: x
        
        # Import and execute the main block
        # Note: This is tricky to test directly, so we test the components
        dataset_func = DATASETS["fw"]
        dataset = dataset_func()
        
        # Simulate the main loop
        samples = list(dataset)
        assert len(samples) == 3
        assert samples == mock_text_data


class TestDatasetErrorHandling:
    """Test error handling in dataset operations."""
    
    @patch('datasets.load_dataset')
    def test_dataset_loading_timeout(self, mock_load_dataset):
        """Test handling of dataset loading timeout."""
        mock_load_dataset.side_effect = TimeoutError("Request timed out")
        
        with pytest.raises(TimeoutError):
            fineweb_10bt_text()
    
    @patch('datasets.load_dataset')
    def test_dataset_loading_permission_error(self, mock_load_dataset):
        """Test handling of dataset loading permission error."""
        mock_load_dataset.side_effect = PermissionError("Access denied")
        
        with pytest.raises(PermissionError):
            fineweb_10bt_text()
    
    @patch('datasets.load_dataset')
    def test_dataset_invalid_name_error(self, mock_load_dataset):
        """Test handling of invalid dataset name."""
        mock_load_dataset.side_effect = ValueError("Dataset not found")
        
        with pytest.raises(ValueError):
            fineweb_10bt_text()
    
    @patch('datasets.load_dataset')
    def test_dataset_missing_text_column(self, mock_load_dataset):
        """Test handling of missing text column."""
        # Mock dataset without text column
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.side_effect = KeyError("text")
        mock_load_dataset.return_value = mock_dataset
        
        with pytest.raises(KeyError):
            fineweb_10bt_text()


class TestDatasetConfiguration:
    """Test dataset configuration and parameters."""
    
    @patch('datasets.load_dataset')
    def test_fineweb_dataset_name(self, mock_load_dataset):
        """Test that correct dataset name is used."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        fineweb_10bt_text()
        
        # Check first positional argument (dataset name)
        call_args = mock_load_dataset.call_args
        assert call_args[0][0] == "HuggingFaceFW/fineweb"
    
    @patch('datasets.load_dataset')
    def test_fineweb_dataset_config(self, mock_load_dataset):
        """Test that correct dataset config is used."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        fineweb_10bt_text()
        
        # Check name parameter
        call_args = mock_load_dataset.call_args
        assert call_args[1]['name'] == "sample-10BT"
    
    @patch('datasets.load_dataset')
    def test_dataset_parameters_completeness(self, mock_load_dataset):
        """Test that all expected parameters are passed to load_dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        fineweb_10bt_text()
        
        call_args = mock_load_dataset.call_args
        expected_kwargs = {'name', 'split', 'streaming'}
        actual_kwargs = set(call_args[1].keys())
        
        assert expected_kwargs.issubset(actual_kwargs), \
            f"Missing parameters: {expected_kwargs - actual_kwargs}"


class TestDatasetTypeAnnotations:
    """Test type annotations and return types."""
    
    @patch('datasets.load_dataset')
    def test_fineweb_return_type_annotation(self, mock_load_dataset):
        """Test that fineweb function has correct return type annotation."""
        import inspect
        from datasets import IterableColumn
        
        # Get function signature
        sig = inspect.signature(fineweb_10bt_text)
        return_annotation = sig.return_annotation
        
        # Should be annotated as IterableColumn
        assert return_annotation == IterableColumn
    
    @patch('datasets.load_dataset')
    def test_dataset_registry_type_consistency(self, mock_load_dataset):
        """Test that dataset registry functions have consistent types."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        # All functions in registry should have callable type
        for name, func in DATASETS.items():
            assert callable(func)
            
            # Should have return type annotation
            import inspect
            sig = inspect.signature(func)
            assert sig.return_annotation != inspect.Signature.empty, \
                f"Function {name} should have return type annotation"

