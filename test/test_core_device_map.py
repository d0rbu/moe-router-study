"""Tests for core.device_map module."""

import pytest

from core.device_map import CUSTOM_DEVICES, MAX_LAYERS, attn_gpu, mlp_gpu


class TestDeviceMapConstants:
    """Test device mapping constants."""
    
    def test_max_layers_value(self):
        """Test that MAX_LAYERS has a reasonable value."""
        assert isinstance(MAX_LAYERS, int)
        assert MAX_LAYERS > 0
        assert MAX_LAYERS == 256  # Current expected value
    
    def test_custom_devices_structure(self):
        """Test CUSTOM_DEVICES dictionary structure."""
        assert isinstance(CUSTOM_DEVICES, dict)
        assert len(CUSTOM_DEVICES) > 0
        
        for name, func in CUSTOM_DEVICES.items():
            assert isinstance(name, str)
            assert callable(func)
    
    def test_custom_devices_content(self):
        """Test that expected device mapping functions are present."""
        expected_functions = ["mlp_gpu", "attn_gpu"]
        
        for func_name in expected_functions:
            assert func_name in CUSTOM_DEVICES
            assert CUSTOM_DEVICES[func_name] is not None


class TestMlpGpu:
    """Test mlp_gpu device mapping function."""
    
    def test_mlp_gpu_returns_dict(self):
        """Test that mlp_gpu returns a dictionary."""
        result = mlp_gpu()
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_mlp_gpu_base_components(self):
        """Test that base model components are mapped to CPU."""
        result = mlp_gpu()
        
        # Base components should be on CPU
        base_components = [
            "model.embed_tokens.weight",
            "model.ln_final.weight", 
            "model.norm.weight",
            "lm_head.weight"
        ]
        
        for component in base_components:
            assert component in result
            assert result[component] == "cpu"
    
    def test_mlp_gpu_attention_components(self):
        """Test that attention components are mapped to CPU."""
        result = mlp_gpu()
        
        # Check a few attention components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                attention_components = [
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                    f"model.layers.{layer_idx}.self_attn.k_norm.weight",
                ]
                
                for component in attention_components:
                    assert component in result
                    assert result[component] == "cpu"
    
    def test_mlp_gpu_mlp_router_components(self):
        """Test that MLP router components are mapped to GPU."""
        result = mlp_gpu()
        
        # Check router components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                router_components = [
                    f"model.layers.{layer_idx}.mlp.router.weight",
                    f"model.layers.{layer_idx}.mlp.gate.weight",
                ]
                
                for component in router_components:
                    assert component in result
                    assert result[component] == 0  # GPU device 0
    
    def test_mlp_gpu_expert_components(self):
        """Test that MLP expert components are mapped to GPU."""
        result = mlp_gpu()
        
        # Check expert components for a few layers and experts
        test_cases = [(0, 0), (1, 5), (10, 100), (50, 511)]
        
        for layer_idx, expert_idx in test_cases:
            if layer_idx < MAX_LAYERS and expert_idx < 512:
                expert_components = [
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                ]
                
                for component in expert_components:
                    assert component in result
                    assert result[component] == 0  # GPU device 0
    
    def test_mlp_gpu_layernorm_components(self):
        """Test that layer norm components are mapped to CPU."""
        result = mlp_gpu()
        
        # Check layer norm components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                layernorm_components = [
                    f"model.layers.{layer_idx}.input_layernorm.weight",
                    f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                ]
                
                for component in layernorm_components:
                    assert component in result
                    assert result[component] == "cpu"
    
    def test_mlp_gpu_all_layers_covered(self):
        """Test that all layers up to MAX_LAYERS are covered."""
        result = mlp_gpu()
        
        # Check that we have entries for all layers
        for layer_idx in range(min(5, MAX_LAYERS)):  # Test first 5 layers
            # At least one component per layer should exist
            layer_components = [
                key for key in result.keys() 
                if f"model.layers.{layer_idx}." in key
            ]
            assert len(layer_components) > 0
    
    def test_mlp_gpu_expert_range(self):
        """Test that experts are covered up to expected range."""
        result = mlp_gpu()
        
        # Check that we have expert components for the expected range
        expert_keys = [
            key for key in result.keys() 
            if "mlp.experts." in key and "gate_proj.weight" in key
        ]
        
        # Should have entries for layer 0, experts 0-511
        layer_0_experts = [
            key for key in expert_keys 
            if key.startswith("model.layers.0.mlp.experts.")
        ]
        
        # Should have 512 experts (0-511)
        assert len(layer_0_experts) == 512


class TestAttnGpu:
    """Test attn_gpu device mapping function."""
    
    def test_attn_gpu_returns_dict(self):
        """Test that attn_gpu returns a dictionary."""
        result = attn_gpu()
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_attn_gpu_base_components(self):
        """Test that base model components are mapped to GPU."""
        result = attn_gpu()
        
        # Base components should be on GPU
        base_components = [
            "model.embed_tokens.weight",
            "model.ln_final.weight",
            "model.norm.weight", 
            "lm_head.weight"
        ]
        
        for component in base_components:
            assert component in result
            assert result[component] == 0  # GPU device 0
    
    def test_attn_gpu_attention_components(self):
        """Test that attention components are mapped to GPU."""
        result = attn_gpu()
        
        # Check attention components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                attention_components = [
                    f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                    f"model.layers.{layer_idx}.self_attn.q_norm.weight",
                    f"model.layers.{layer_idx}.self_attn.k_norm.weight",
                ]
                
                for component in attention_components:
                    assert component in result
                    assert result[component] == 0  # GPU device 0
    
    def test_attn_gpu_mlp_router_components(self):
        """Test that MLP router components are mapped to GPU."""
        result = attn_gpu()
        
        # Check router components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                router_components = [
                    f"model.layers.{layer_idx}.mlp.router.weight",
                    f"model.layers.{layer_idx}.mlp.gate.weight",
                ]
                
                for component in router_components:
                    assert component in result
                    assert result[component] == 0  # GPU device 0
    
    def test_attn_gpu_expert_components(self):
        """Test that MLP expert components are mapped to CPU."""
        result = attn_gpu()
        
        # Check expert components for a few layers and experts
        test_cases = [(0, 0), (1, 5), (10, 100), (50, 511)]
        
        for layer_idx, expert_idx in test_cases:
            if layer_idx < MAX_LAYERS and expert_idx < 512:
                expert_components = [
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", 
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                ]
                
                for component in expert_components:
                    assert component in result
                    assert result[component] == "cpu"
    
    def test_attn_gpu_layernorm_components(self):
        """Test that layer norm components are mapped to GPU."""
        result = attn_gpu()
        
        # Check layer norm components for different layers
        for layer_idx in [0, 1, 10, 50]:
            if layer_idx < MAX_LAYERS:
                layernorm_components = [
                    f"model.layers.{layer_idx}.input_layernorm.weight",
                    f"model.layers.{layer_idx}.post_attention_layernorm.weight",
                ]
                
                for component in layernorm_components:
                    assert component in result
                    assert result[component] == 0  # GPU device 0


class TestDeviceMappingComparison:
    """Test comparisons between different device mapping strategies."""
    
    def test_mlp_vs_attn_gpu_differences(self):
        """Test the key differences between mlp_gpu and attn_gpu."""
        mlp_result = mlp_gpu()
        attn_result = attn_gpu()
        
        # Both should have the same keys
        assert set(mlp_result.keys()) == set(attn_result.keys())
        
        # Base components: mlp_gpu -> CPU, attn_gpu -> GPU
        base_components = [
            "model.embed_tokens.weight",
            "model.ln_final.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]
        
        for component in base_components:
            assert mlp_result[component] == "cpu"
            assert attn_result[component] == 0
        
        # Attention components: mlp_gpu -> CPU, attn_gpu -> GPU
        attn_component = "model.layers.0.self_attn.q_proj.weight"
        assert mlp_result[attn_component] == "cpu"
        assert attn_result[attn_component] == 0
        
        # Expert components: mlp_gpu -> GPU, attn_gpu -> CPU
        expert_component = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert mlp_result[expert_component] == 0
        assert attn_result[expert_component] == "cpu"
        
        # Router components: both -> GPU
        router_component = "model.layers.0.mlp.router.weight"
        assert mlp_result[router_component] == 0
        assert attn_result[router_component] == 0
    
    def test_device_mapping_completeness(self):
        """Test that device mappings cover all expected component types."""
        mlp_result = mlp_gpu()
        
        # Check that we have all expected component types
        component_types = {
            "embed_tokens": False,
            "ln_final": False,
            "norm": False,
            "lm_head": False,
            "q_proj": False,
            "k_proj": False,
            "v_proj": False,
            "o_proj": False,
            "q_norm": False,
            "k_norm": False,
            "router": False,
            "gate": False,
            "gate_proj": False,
            "up_proj": False,
            "down_proj": False,
            "input_layernorm": False,
            "post_attention_layernorm": False,
        }
        
        for key in mlp_result.keys():
            for component_type in component_types.keys():
                if component_type in key:
                    component_types[component_type] = True
        
        # All component types should be present
        missing_types = [t for t, found in component_types.items() if not found]
        assert len(missing_types) == 0, f"Missing component types: {missing_types}"


class TestCustomDevicesIntegration:
    """Test integration with CUSTOM_DEVICES dictionary."""
    
    def test_custom_devices_functions_work(self):
        """Test that functions in CUSTOM_DEVICES can be called."""
        for name, func in CUSTOM_DEVICES.items():
            result = func()
            assert isinstance(result, dict)
            assert len(result) > 0
    
    def test_custom_devices_mlp_gpu(self):
        """Test that CUSTOM_DEVICES['mlp_gpu'] works correctly."""
        result = CUSTOM_DEVICES["mlp_gpu"]()
        expected = mlp_gpu()
        assert result == expected
    
    def test_custom_devices_attn_gpu(self):
        """Test that CUSTOM_DEVICES['attn_gpu'] works correctly."""
        result = CUSTOM_DEVICES["attn_gpu"]()
        expected = attn_gpu()
        assert result == expected
    
    def test_device_mapping_consistency(self):
        """Test that device mappings are internally consistent."""
        for name, func in CUSTOM_DEVICES.items():
            device_map = func()
            
            # All keys should be strings
            for key in device_map.keys():
                assert isinstance(key, str)
                assert len(key) > 0
            
            # All values should be valid device specifications
            for value in device_map.values():
                assert value in ["cpu", 0] or isinstance(value, int)


class TestDeviceMappingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_max_layer_boundary(self):
        """Test behavior at MAX_LAYERS boundary."""
        mlp_result = mlp_gpu()
        
        # Should have components for layer MAX_LAYERS - 1
        last_layer_key = f"model.layers.{MAX_LAYERS - 1}.self_attn.q_proj.weight"
        assert last_layer_key in mlp_result
        
        # Should NOT have components for layer MAX_LAYERS
        beyond_max_key = f"model.layers.{MAX_LAYERS}.self_attn.q_proj.weight"
        assert beyond_max_key not in mlp_result
    
    def test_expert_range_boundary(self):
        """Test expert range boundaries."""
        mlp_result = mlp_gpu()
        
        # Should have expert 511 (last expert)
        last_expert_key = "model.layers.0.mlp.experts.511.gate_proj.weight"
        assert last_expert_key in mlp_result
        
        # Should NOT have expert 512
        beyond_expert_key = "model.layers.0.mlp.experts.512.gate_proj.weight"
        assert beyond_expert_key not in mlp_result
    
    def test_device_map_size(self):
        """Test that device maps have reasonable sizes."""
        mlp_result = mlp_gpu()
        attn_result = attn_gpu()
        
        # Should have a substantial number of entries
        assert len(mlp_result) > 1000  # Many layers × many experts
        assert len(attn_result) > 1000
        
        # Both should have the same number of entries
        assert len(mlp_result) == len(attn_result)
