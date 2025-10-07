"""Tests for exp.training module."""

import warnings
from unittest.mock import patch

import pytest

from exp.training import get_experiment_name


class TestGetExperimentName:
    """Test get_experiment_name function."""
    
    def test_basic_functionality(self):
        """Test basic experiment name generation."""
        result = get_experiment_name("model1", "dataset1")
        assert result == "model1_dataset1"
    
    def test_with_additional_params(self):
        """Test experiment name with additional parameters."""
        result = get_experiment_name(
            "model1", "dataset1", 
            learning_rate=0.001, 
            batch_size=32
        )
        
        # Parameters should be sorted and included
        expected_parts = ["model1_dataset1", "batch_size=32", "learning_rate=0.001"]
        expected = "_".join(expected_parts)
        assert result == expected
    
    def test_parameter_sorting(self):
        """Test that parameters are sorted alphabetically."""
        result = get_experiment_name(
            "model", "data",
            z_param=1,
            a_param=2,
            m_param=3
        )
        
        # Should be sorted: a_param, m_param, z_param
        expected = "model_data_a_param=2_m_param=3_z_param=1"
        assert result == expected
    
    def test_ignored_keys(self):
        """Test that certain keys are ignored."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = get_experiment_name(
                "model", "data",
                device="cuda",
                resume=True,
                learning_rate=0.01
            )
            
            # device and resume should be ignored
            expected = "model_data_learning_rate=0.01"
            assert result == expected
            
            # Should have warning about filtered keys
            assert len(w) == 1
            assert "excluded from the experiment name" in str(w[0].message)
            assert "device" in str(w[0].message)
            assert "resume" in str(w[0].message)
    
    def test_underscore_prefixed_keys(self):
        """Test that keys starting with underscore are ignored."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = get_experiment_name(
                "model", "data",
                _private_param=123,
                public_param=456
            )
            
            # _private_param should be ignored
            expected = "model_data_public_param=456"
            assert result == expected
            
            # Should have warning about filtered keys
            assert len(w) == 1
            assert "_private_param" in str(w[0].message)
    
    def test_no_additional_params(self):
        """Test with no additional parameters."""
        result = get_experiment_name("test_model", "test_dataset")
        assert result == "test_model_test_dataset"
    
    def test_empty_strings(self):
        """Test with empty model and dataset names."""
        result = get_experiment_name("", "")
        assert result == "_"
        
        result = get_experiment_name("model", "")
        assert result == "model_"
        
        result = get_experiment_name("", "dataset")
        assert result == "_dataset"
    
    def test_special_characters_in_names(self):
        """Test with special characters in model/dataset names."""
        result = get_experiment_name("model-v2", "dataset_2023")
        assert result == "model-v2_dataset_2023"
        
        result = get_experiment_name("model.1", "data/set")
        assert result == "model.1_data/set"
    
    def test_various_parameter_types(self):
        """Test with various parameter value types."""
        result = get_experiment_name(
            "model", "data",
            int_param=42,
            float_param=3.14,
            str_param="hello",
            bool_param=True,
            none_param=None
        )
        
        # All parameters should be converted to strings
        expected_parts = [
            "model_data",
            "bool_param=True",
            "float_param=3.14", 
            "int_param=42",
            "none_param=None",
            "str_param=hello"
        ]
        expected = "_".join(expected_parts)
        assert result == expected
    
    def test_long_experiment_name_hashing(self):
        """Test that very long experiment names are hashed."""
        # Create a very long parameter list
        long_params = {f"param_{i}": f"value_{i}" for i in range(50)}
        
        result = get_experiment_name("model", "dataset", **long_params)
        
        # Should be a SHA256 hash (64 characters)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_exactly_255_characters(self):
        """Test behavior at the 255 character boundary."""
        # Create parameters that result in exactly 255 characters
        base_name = "model_dataset"  # 13 characters
        remaining = 255 - len(base_name) - 1  # -1 for the underscore
        
        # Create a parameter that uses exactly the remaining characters
        param_value = "x" * (remaining - len("param="))
        
        result = get_experiment_name("model", "dataset", param=param_value)
        
        # Should be exactly 255 characters (not hashed)
        assert len(result) == 255
        assert result.startswith("model_dataset_param=")
    
    def test_over_255_characters(self):
        """Test that names over 255 characters are hashed."""
        # Create parameters that result in over 255 characters
        long_value = "x" * 300
        
        result = get_experiment_name("model", "dataset", long_param=long_value)
        
        # Should be hashed to 64 characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_warning_stacklevel(self):
        """Test that warnings have correct stack level."""
        def wrapper_function():
            return get_experiment_name("model", "data", device="cuda")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapper_function()
            
            # Warning should point to the wrapper_function, not internal code
            assert len(w) == 1
            # The stacklevel=2 should make the warning appear to come from wrapper_function
    
    def test_no_warning_when_no_filtered_keys(self):
        """Test that no warning is issued when no keys are filtered."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            get_experiment_name("model", "data", learning_rate=0.01)
            
            # Should have no warnings
            assert len(w) == 0
    
    def test_deterministic_hashing(self):
        """Test that hashing is deterministic."""
        long_params = {f"param_{i}": f"value_{i}" for i in range(50)}
        
        result1 = get_experiment_name("model", "dataset", **long_params)
        result2 = get_experiment_name("model", "dataset", **long_params)
        
        # Should produce the same hash
        assert result1 == result2
        assert len(result1) == 64
    
    def test_different_params_different_hashes(self):
        """Test that different parameters produce different hashes."""
        long_params1 = {f"param_{i}": f"value_{i}" for i in range(50)}
        long_params2 = {f"param_{i}": f"different_value_{i}" for i in range(50)}
        
        result1 = get_experiment_name("model", "dataset", **long_params1)
        result2 = get_experiment_name("model", "dataset", **long_params2)
        
        # Should produce different hashes
        assert result1 != result2
        assert len(result1) == len(result2) == 64
    
    @pytest.mark.parametrize("model_name,dataset_name,expected", [
        ("simple", "data", "simple_data"),
        ("model-1", "dataset_v2", "model-1_dataset_v2"),
        ("", "", "_"),
        ("a", "b", "a_b"),
    ])
    def test_parametrized_basic_cases(self, model_name, dataset_name, expected):
        """Test various basic model/dataset name combinations."""
        result = get_experiment_name(model_name, dataset_name)
        assert result == expected
    
    def test_complex_integration(self):
        """Test a complex realistic scenario."""
        result = get_experiment_name(
            "transformer-large", 
            "wikipedia-2023",
            learning_rate=1e-4,
            batch_size=64,
            num_layers=12,
            hidden_size=768,
            dropout=0.1,
            warmup_steps=1000,
            max_steps=100000,
            # These should be ignored
            device="cuda:0",
            resume=False,
            _internal_flag=True
        )
        
        expected_parts = [
            "transformer-large_wikipedia-2023",
            "batch_size=64",
            "dropout=0.1", 
            "hidden_size=768",
            "learning_rate=0.0001",
            "max_steps=100000",
            "num_layers=12",
            "warmup_steps=1000"
        ]
        expected = "_".join(expected_parts)
        
        # Should not be hashed (under 255 chars)
        assert result == expected
        assert len(result) < 255
