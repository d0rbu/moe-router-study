"""Tests for core.device_map module."""

import pytest

from core.device_map import MAX_LAYERS, CUSTOM_DEVICES, attn_gpu, mlp_gpu
from test.test_utils import validate_device_map


class TestDeviceMapFunctions:
    """Test device mapping functions."""
    
    def test_mlp_gpu_basic_structure(self):
        """Test that mlp_gpu returns a properly structured device map."""
        device_map = mlp_gpu()
        
        # Validate basic structure
        validate_device_map(device_map, ["cpu", 0])
        
        # Check that base components are on CPU
        assert device_map["model.embed_tokens.weight"] == "cpu"
        assert device_map["model.ln_final.weight"] == "cpu"
        assert device_map["model.norm.weight"] == "cpu"
        assert device_map["lm_head.weight"] == "cpu"
    
    def test_mlp_gpu_attention_on_cpu(self):
        """Test that attention components are on CPU in mlp_gpu."""
        device_map = mlp_gpu()
        
        # Check a few attention components are on CPU
        assert device_map["model.layers.0.self_attn.q_proj.weight"] == "cpu"
        assert device_map["model.layers.0.self_attn.k_proj.weight"] == "cpu"
        assert device_map["model.layers.0.self_attn.v_proj.weight"] == "cpu"
        assert device_map["model.layers.0.self_attn.o_proj.weight"] == "cpu"
    
    def test_mlp_gpu_mlp_on_gpu(self):
        """Test that MLP components are on GPU in mlp_gpu."""
        device_map = mlp_gpu()
        
        # Check MLP router and gate components are on GPU
        assert device_map["model.layers.0.mlp.router.weight"] == 0
        assert device_map["model.layers.0.mlp.gate.weight"] == 0
        
        # Check expert components are on GPU
        assert device_map["model.layers.0.mlp.experts.0.gate_proj.weight"] == 0
        assert device_map["model.layers.0.mlp.experts.0.up_proj.weight"] == 0
        assert device_map["model.layers.0.mlp.experts.0.down_proj.weight"] == 0
    
    def test_attn_gpu_basic_structure(self):
        """Test that attn_gpu returns a properly structured device map."""
        device_map = attn_gpu()
        
        # Validate basic structure
        validate_device_map(device_map, ["cpu", 0])
        
        # Check that base components are on GPU
        assert device_map["model.embed_tokens.weight"] == 0
        assert device_map["model.ln_final.weight"] == 0
        assert device_map["model.norm.weight"] == 0
        assert device_map["lm_head.weight"] == 0
    
    def test_attn_gpu_attention_on_gpu(self):
        """Test that attention components are on GPU in attn_gpu."""
        device_map = attn_gpu()
        
        # Check attention components are on GPU
        assert device_map["model.layers.0.self_attn.q_proj.weight"] == 0
        assert device_map["model.layers.0.self_attn.k_proj.weight"] == 0
        assert device_map["model.layers.0.self_attn.v_proj.weight"] == 0
        assert device_map["model.layers.0.self_attn.o_proj.weight"] == 0
    
    def test_attn_gpu_mlp_experts_on_cpu(self):
        """Test that MLP expert components are on CPU in attn_gpu."""
        device_map = attn_gpu()
        
        # Check expert components are on CPU
        assert device_map["model.layers.0.mlp.experts.0.gate_proj.weight"] == "cpu"
        assert device_map["model.layers.0.mlp.experts.0.up_proj.weight"] == "cpu"
        assert device_map["model.layers.0.mlp.experts.0.down_proj.weight"] == "cpu"
        
        # But router and gate should still be on GPU
        assert device_map["model.layers.0.mlp.router.weight"] == 0
        assert device_map["model.layers.0.mlp.gate.weight"] == 0
    
    def test_device_map_completeness(self):
        """Test that device maps cover all expected layers."""
        mlp_map = mlp_gpu()
        attn_map = attn_gpu()
        
        # Both should have the same keys
        assert set(mlp_map.keys()) == set(attn_map.keys())
        
        # Should cover all layers up to MAX_LAYERS
        layer_keys = [key for key in mlp_map.keys() if "model.layers." in key]
        
        # Extract layer indices
        layer_indices = set()
        for key in layer_keys:
            parts = key.split(".")
            if len(parts) >= 3 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
        
        # Should have entries for layers 0 through MAX_LAYERS-1
        expected_indices = set(range(MAX_LAYERS))
        assert layer_indices == expected_indices
    
    def test_expert_coverage(self):
        """Test that device maps cover all experts."""
        device_map = mlp_gpu()
        
        # Check that we have expert entries for multiple experts
        expert_keys = [key for key in device_map.keys() if "experts." in key]
        
        # Extract expert indices
        expert_indices = set()
        for key in expert_keys:
            parts = key.split(".")
            expert_part_idx = None
            for i, part in enumerate(parts):
                if part == "experts" and i + 1 < len(parts):
                    expert_part_idx = i + 1
                    break
            
            if expert_part_idx and parts[expert_part_idx].isdigit():
                expert_indices.add(int(parts[expert_part_idx]))
        
        # Should have experts 0 through 511 (512 experts total)
        assert len(expert_indices) == 512
        assert min(expert_indices) == 0
        assert max(expert_indices) == 511
    
    def test_layernorm_components(self):
        """Test that layernorm components are properly mapped."""
        mlp_map = mlp_gpu()
        attn_map = attn_gpu()
        
        # Check layernorm components exist and have proper device assignment
        for layer_idx in range(min(5, MAX_LAYERS)):  # Test first 5 layers
            input_ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            post_attn_ln_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            
            assert input_ln_key in mlp_map
            assert post_attn_ln_key in mlp_map
            assert input_ln_key in attn_map
            assert post_attn_ln_key in attn_map
            
            # In mlp_gpu, layernorms should be on CPU
            assert mlp_map[input_ln_key] == "cpu"
            assert mlp_map[post_attn_ln_key] == "cpu"
            
            # In attn_gpu, layernorms should be on GPU
            assert attn_map[input_ln_key] == 0
            assert attn_map[post_attn_ln_key] == 0


class TestCustomDevicesRegistry:
    """Test the CUSTOM_DEVICES registry."""
    
    def test_registry_structure(self):
        """Test that CUSTOM_DEVICES registry is properly structured."""
        assert isinstance(CUSTOM_DEVICES, dict)
        assert "mlp_gpu" in CUSTOM_DEVICES
        assert "attn_gpu" in CUSTOM_DEVICES
    
    def test_registry_functions_callable(self):
        """Test that registry functions are callable."""
        for name, func in CUSTOM_DEVICES.items():
            assert callable(func), f"Function {name} should be callable"
    
    def test_registry_functions_return_dict(self):
        """Test that registry functions return dictionaries."""
        for name, func in CUSTOM_DEVICES.items():
            result = func()
            assert isinstance(result, dict), f"Function {name} should return a dict"
            assert len(result) > 0, f"Function {name} should return non-empty dict"
    
    def test_registry_consistency(self):
        """Test that registry functions return consistent results."""
        # Call each function multiple times and ensure consistent results
        for name, func in CUSTOM_DEVICES.items():
            result1 = func()
            result2 = func()
            assert result1 == result2, f"Function {name} should return consistent results"


class TestDeviceMapEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_max_layers_constant(self):
        """Test that MAX_LAYERS is a reasonable value."""
        assert isinstance(MAX_LAYERS, int)
        assert MAX_LAYERS > 0
        assert MAX_LAYERS <= 1024  # Reasonable upper bound
    
    def test_device_map_keys_are_strings(self):
        """Test that all device map keys are strings."""
        for func_name, func in CUSTOM_DEVICES.items():
            device_map = func()
            for key in device_map.keys():
                assert isinstance(key, str), f"Key {key} in {func_name} should be string"
    
    def test_device_map_values_are_valid(self):
        """Test that all device map values are valid device specifications."""
        valid_devices = {"cpu", 0}
        
        for func_name, func in CUSTOM_DEVICES.items():
            device_map = func()
            for key, device in device_map.items():
                assert device in valid_devices, \
                    f"Invalid device {device} for key {key} in {func_name}"
    
    def test_no_duplicate_keys(self):
        """Test that device maps don't have duplicate keys."""
        for func_name, func in CUSTOM_DEVICES.items():
            device_map = func()
            keys = list(device_map.keys())
            unique_keys = set(keys)
            assert len(keys) == len(unique_keys), \
                f"Duplicate keys found in {func_name}: {set(keys) - unique_keys}"


class TestDeviceMapIntegration:
    """Integration tests for device mapping."""
    
    def test_device_maps_cover_same_components(self):
        """Test that different device maps cover the same model components."""
        mlp_map = mlp_gpu()
        attn_map = attn_gpu()
        
        # Should have identical key sets
        assert set(mlp_map.keys()) == set(attn_map.keys())
    
    def test_device_assignment_differences(self):
        """Test that device maps have expected differences in device assignments."""
        mlp_map = mlp_gpu()
        attn_map = attn_gpu()
        
        differences = 0
        for key in mlp_map.keys():
            if mlp_map[key] != attn_map[key]:
                differences += 1
        
        # Should have many differences (different device assignments)
        assert differences > 0, "Device maps should have different device assignments"
        
        # But not all keys should be different (some components might be on same device)
        assert differences < len(mlp_map), "Not all components should have different assignments"
    
    def test_complementary_gpu_usage(self):
        """Test that mlp_gpu and attn_gpu use GPU for complementary components."""
        mlp_map = mlp_gpu()
        attn_map = attn_gpu()
        
        # Count GPU assignments in each map
        mlp_gpu_count = sum(1 for device in mlp_map.values() if device == 0)
        attn_gpu_count = sum(1 for device in attn_map.values() if device == 0)
        
        # Both should use GPU, but for different components
        assert mlp_gpu_count > 0, "mlp_gpu should assign some components to GPU"
        assert attn_gpu_count > 0, "attn_gpu should assign some components to GPU"
        
        # The total GPU usage should be substantial but not identical
        assert mlp_gpu_count != attn_gpu_count, "GPU usage should differ between maps"

