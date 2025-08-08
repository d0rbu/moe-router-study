"""Basic tests to verify the setup works."""

import pytest
import torch as th


def test_imports() -> None:
    """Test that basic imports work."""
    import core
    import exp
    import viz

    assert core.__version__ == "0.1.0"
    assert hasattr(exp, 'OUTPUT_DIR')
    assert hasattr(viz, 'FIGURE_DIR')


def test_basic_functionality() -> None:
    """Test basic functionality."""
    assert 1 + 1 == 2


def test_torch_functionality() -> None:
    """Test that PyTorch works correctly."""
    # Test basic tensor operations
    x = th.tensor([1.0, 2.0, 3.0])
    y = th.tensor([4.0, 5.0, 6.0])
    z = x + y
    
    expected = th.tensor([5.0, 7.0, 9.0])
    assert th.allclose(z, expected)
    
    # Test boolean operations
    mask = x > 1.5
    expected_mask = th.tensor([False, True, True])
    assert th.equal(mask, expected_mask)


def test_core_module_structure() -> None:
    """Test that core module has expected structure."""
    import core.data
    import core.device_map
    import core.model
    
    # Test that key components exist
    assert hasattr(core.data, 'DATASETS')
    assert hasattr(core.device_map, 'CUSTOM_DEVICES')
    assert hasattr(core.model, 'MODELS')
    
    # Test that they are the right types
    assert isinstance(core.data.DATASETS, dict)
    assert isinstance(core.device_map.CUSTOM_DEVICES, dict)
    assert isinstance(core.model.MODELS, dict)


def test_exp_module_structure() -> None:
    """Test that exp module has expected structure."""
    import exp.activations
    import exp.circuit_loss
    import exp.circuit_optimization
    
    # Test that key functions exist
    assert hasattr(exp.activations, 'load_activations')
    assert hasattr(exp.circuit_loss, 'circuit_loss')
    assert hasattr(exp.circuit_optimization, 'expand_batch')
    
    # Test that they are callable
    assert callable(exp.activations.load_activations)
    assert callable(exp.circuit_loss.circuit_loss)
    assert callable(exp.circuit_optimization.expand_batch)


def test_viz_module_structure() -> None:
    """Test that viz module has expected structure."""
    import viz.pca_circuits
    import viz.router_correlations
    import viz.router_spaces
    
    # Test that key functions exist
    assert hasattr(viz.pca_circuits, 'pca_figure')
    assert hasattr(viz.router_correlations, 'router_correlations')
    assert hasattr(viz.router_spaces, 'router_spaces')
    
    # Test that they are callable
    assert callable(viz.pca_circuits.pca_figure)
    assert callable(viz.router_correlations.router_correlations)
    assert callable(viz.router_spaces.router_spaces)


@pytest.mark.slow
def test_nnterp_import() -> None:
    """Test that nnterp can be imported (marked as slow since it's a heavy import)."""
    import nnterp

    assert hasattr(nnterp, "StandardizedTransformer")


@pytest.mark.slow
def test_heavy_dependencies() -> None:
    """Test that heavy dependencies can be imported."""
    # Test transformers
    import transformers
    assert hasattr(transformers, 'AutoModelForCausalLM')
    assert hasattr(transformers, 'AutoTokenizer')
    
    # Test datasets
    import datasets
    assert hasattr(datasets, 'load_dataset')
    
    # Test matplotlib
    import matplotlib.pyplot as plt
    assert hasattr(plt, 'scatter')
    assert hasattr(plt, 'savefig')


def test_device_availability() -> None:
    """Test device availability and tensor operations."""
    # Test CPU operations
    x = th.tensor([1.0, 2.0], device='cpu')
    assert x.device.type == 'cpu'
    
    # Test CUDA availability (but don't require it)
    cuda_available = th.cuda.is_available()
    if cuda_available:
        # If CUDA is available, test basic operations
        try:
            x_cuda = th.tensor([1.0, 2.0], device='cuda')
            assert x_cuda.device.type == 'cuda'
            
            # Test moving between devices
            x_back_to_cpu = x_cuda.cpu()
            assert x_back_to_cpu.device.type == 'cpu'
        except RuntimeError:
            # CUDA might be available but not functional
            pass


def test_mathematical_operations() -> None:
    """Test mathematical operations used in the codebase."""
    # Test topk operation
    x = th.tensor([[3.0, 1.0, 4.0, 2.0]])
    topk_values, topk_indices = th.topk(x, k=2, dim=1)
    
    expected_values = th.tensor([[4.0, 3.0]])
    expected_indices = th.tensor([[2, 0]])
    
    assert th.allclose(topk_values, expected_values)
    assert th.equal(topk_indices, expected_indices)
    
    # Test boolean scatter operation
    target = th.zeros(1, 4, dtype=th.bool)
    target.scatter_(1, topk_indices, True)
    
    expected_target = th.tensor([[True, False, True, False]])
    assert th.equal(target, expected_target)
    
    # Test IoU-like operations
    a = th.tensor([[True, False, True, False]], dtype=th.bool)
    b = th.tensor([[True, True, False, False]], dtype=th.bool)
    
    intersection = (a & b).sum()
    union = (a | b).sum()
    iou = intersection.float() / union.float()
    
    assert intersection == 1  # Only first position overlaps
    assert union == 3  # Three positions have at least one True
    assert abs(iou - 1.0/3.0) < 1e-6


def test_file_operations() -> None:
    """Test file operations used in the codebase."""
    import tempfile
    import os
    from pathlib import Path
    
    # Test temporary file creation and tensor saving/loading
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test tensor save/load
        test_tensor = th.randn(5, 3)
        test_file = temp_path / "test.pt"
        
        th.save(test_tensor, test_file)
        assert test_file.exists()
        
        loaded_tensor = th.load(test_file)
        assert th.allclose(test_tensor, loaded_tensor)
        
        # Test directory operations
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir(exist_ok=True)
        assert sub_dir.exists()
        assert sub_dir.is_dir()


def test_error_handling_patterns() -> None:
    """Test common error handling patterns."""
    # Test assertion errors
    with pytest.raises(AssertionError):
        assert False, "This should raise an assertion error"
    
    # Test value errors
    with pytest.raises(ValueError):
        th.topk(th.tensor([1.0, 2.0]), k=5)  # k > tensor size
    
    # Test key errors
    test_dict = {'a': 1, 'b': 2}
    with pytest.raises(KeyError):
        _ = test_dict['c']
    
    # Test type errors
    with pytest.raises(TypeError):
        th.tensor([1, 2, 3]) + "string"  # Can't add tensor and string
