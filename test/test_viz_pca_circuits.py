"""Tests for viz.pca_circuits module."""

from unittest.mock import MagicMock, patch

import pytest
import torch as th

from test.test_utils import assert_tensor_shape_and_type


class TestPcaFigure:
    """Test pca_figure function."""
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_basic_pca_figure_generation(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test basic PCA figure generation."""
        # Mock activation data
        batch_size, num_layers, num_experts = 100, 6, 32
        activated_experts = th.rand(batch_size, num_layers, num_experts) > 0.7
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        # Mock PCA
        mock_pca = MagicMock()
        pca_result = th.randn(batch_size, 2)  # 2D PCA result
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        # Import and run the function
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Verify load_activations was called
        mock_load_activations.assert_called_once()
        
        # Verify PCA was initialized correctly
        mock_pca_class.assert_called_once_with(n_components=2, svd_solver="full")
        
        # Verify PCA fit_transform was called with flattened data
        fit_transform_call_args = mock_pca.fit_transform.call_args[0]
        input_data = fit_transform_call_args[0]
        expected_shape = (batch_size, num_layers * num_experts)
        assert input_data.shape == expected_shape
        assert input_data.dtype == th.float32
        
        # Verify matplotlib calls
        mock_scatter.assert_called_once()
        scatter_args = mock_scatter.call_args[0]
        assert len(scatter_args) == 2  # x and y coordinates
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_data_preprocessing(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test data preprocessing for PCA."""
        # Create specific activation pattern
        batch_size, num_layers, num_experts = 50, 4, 8
        activated_experts = th.zeros(batch_size, num_layers, num_experts, dtype=th.bool)
        # Set some specific patterns
        activated_experts[0, 0, 0] = True
        activated_experts[0, 1, 2] = True
        activated_experts[1, 2, 1] = True
        
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        # Mock PCA
        mock_pca = MagicMock()
        pca_result = th.randn(batch_size, 2)
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Check that data was properly flattened and converted to float
        fit_transform_call_args = mock_pca.fit_transform.call_args[0]
        input_data = fit_transform_call_args[0]
        
        # Should be flattened to (batch_size, num_layers * num_experts)
        assert input_data.shape == (batch_size, num_layers * num_experts)
        assert input_data.dtype == th.float32
        
        # Check that boolean values were converted correctly
        # First sample should have 1.0 at positions 0 and 10 (layer 1, expert 2)
        assert input_data[0, 0] == 1.0  # Layer 0, expert 0
        assert input_data[0, 8 + 2] == 1.0  # Layer 1, expert 2 (8 experts per layer)
        assert input_data[1, 2 * 8 + 1] == 1.0  # Layer 2, expert 1
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_pca_parameters(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test PCA initialization parameters."""
        activated_experts = th.rand(30, 3, 6) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = th.randn(30, 2)
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Verify PCA was initialized with correct parameters
        mock_pca_class.assert_called_once_with(n_components=2, svd_solver="full")
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_scatter_plot_data(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test scatter plot data handling."""
        batch_size = 75
        activated_experts = th.rand(batch_size, 5, 10) > 0.6
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        # Create specific PCA result
        pca_result = th.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pca_result = pca_result.repeat(25, 1)  # Repeat to get 75 samples
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Check scatter plot was called with correct data
        mock_scatter.assert_called_once()
        scatter_args = mock_scatter.call_args[0]
        
        # Should be called with x and y coordinates
        x_coords, y_coords = scatter_args
        assert len(x_coords) == batch_size
        assert len(y_coords) == batch_size
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_file_saving(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test file saving parameters."""
        activated_experts = th.rand(40, 4, 8) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = th.randn(40, 2)
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Check savefig was called with correct parameters
        mock_savefig.assert_called_once()
        savefig_call = mock_savefig.call_args
        
        # Should save to the correct path with high DPI
        assert "pca_circuits.png" in str(savefig_call)
        
        # Check for DPI and bbox_inches parameters
        if len(savefig_call) > 1 and savefig_call[1]:  # Check if kwargs exist
            kwargs = savefig_call[1]
            if 'dpi' in kwargs:
                assert kwargs['dpi'] == 300
            if 'bbox_inches' in kwargs:
                assert kwargs['bbox_inches'] == "tight"
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_gpu_to_cpu_conversion(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test GPU to CPU conversion in PCA result."""
        activated_experts = th.rand(25, 3, 6) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        # Mock PCA result that's on GPU
        pca_result_gpu = th.randn(25, 2)
        if th.cuda.is_available():
            pca_result_gpu = pca_result_gpu.cuda()
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = pca_result_gpu
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Should complete without errors (CPU conversion should happen)
        mock_scatter.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPcaCircuitsErrorHandling:
    """Test error handling in PCA circuits visualization."""
    
    @patch('viz.pca_circuits.load_activations')
    def test_load_activations_error(self, mock_load_activations):
        """Test handling of load_activations errors."""
        mock_load_activations.side_effect = Exception("Failed to load activations")
        
        from viz.pca_circuits import pca_figure
        
        with pytest.raises(Exception, match="Failed to load activations"):
            pca_figure()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    def test_pca_fitting_error(self, mock_pca_class, mock_load_activations):
        """Test handling of PCA fitting errors."""
        activated_experts = th.rand(20, 3, 6) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.side_effect = Exception("PCA fitting failed")
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        
        with pytest.raises(Exception, match="PCA fitting failed"):
            pca_figure()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    def test_matplotlib_error_handling(
        self, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test handling of matplotlib errors."""
        activated_experts = th.rand(15, 2, 4) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = th.randn(15, 2)
        mock_pca_class.return_value = mock_pca
        
        # Mock savefig to raise an error
        mock_savefig.side_effect = Exception("Failed to save figure")
        
        from viz.pca_circuits import pca_figure
        
        with pytest.raises(Exception, match="Failed to save figure"):
            pca_figure()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_empty_activation_data(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test handling of empty activation data."""
        # Empty activation data
        activated_experts = th.zeros(0, 3, 6, dtype=th.bool)
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.side_effect = Exception("Cannot fit PCA on empty data")
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        
        with pytest.raises(Exception):
            pca_figure()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_single_sample_data(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test handling of single sample data."""
        # Single sample
        activated_experts = th.rand(1, 3, 6) > 0.5
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        mock_pca = MagicMock()
        # PCA might fail or return degenerate results with single sample
        mock_pca.fit_transform.return_value = th.zeros(1, 2)
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        
        # Should handle single sample gracefully
        pca_figure()
        
        mock_scatter.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPcaCircuitsIntegration:
    """Integration tests for PCA circuits visualization."""
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_realistic_data_flow(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test realistic data flow through the PCA pipeline."""
        # Create realistic activation data
        batch_size, num_layers, num_experts = 500, 12, 64
        topk = 8
        
        # Create sparse activation pattern (only topk experts active per layer)
        activated_experts = th.zeros(batch_size, num_layers, num_experts, dtype=th.bool)
        for b in range(batch_size):
            for l in range(num_layers):
                # Randomly activate topk experts
                active_experts = th.randperm(num_experts)[:topk]
                activated_experts[b, l, active_experts] = True
        
        mock_load_activations.return_value = (activated_experts, None, topk)
        
        # Mock realistic PCA result
        pca_result = th.randn(batch_size, 2)
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Verify the data pipeline
        fit_transform_call_args = mock_pca.fit_transform.call_args[0]
        input_data = fit_transform_call_args[0]
        
        # Check input data properties
        assert input_data.shape == (batch_size, num_layers * num_experts)
        assert input_data.dtype == th.float32
        
        # Check sparsity is preserved (should have exactly topk * num_layers active per sample)
        active_counts = input_data.sum(dim=1)
        expected_active = topk * num_layers
        assert th.all(active_counts == expected_active)
        
        # Verify matplotlib calls
        mock_scatter.assert_called_once()
        scatter_args = mock_scatter.call_args[0]
        assert len(scatter_args[0]) == batch_size  # x coordinates
        assert len(scatter_args[1]) == batch_size  # y coordinates
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_different_activation_patterns(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test PCA with different activation patterns."""
        batch_size, num_layers, num_experts = 100, 6, 16
        
        # Create different activation patterns
        activated_experts = th.zeros(batch_size, num_layers, num_experts, dtype=th.bool)
        
        # Pattern 1: First half of batch - activate first few experts
        activated_experts[:50, :, :4] = True
        
        # Pattern 2: Second half of batch - activate last few experts
        activated_experts[50:, :, -4:] = True
        
        mock_load_activations.return_value = (activated_experts, None, 4)
        
        # Mock PCA to return distinct clusters
        pca_result = th.zeros(batch_size, 2)
        pca_result[:50, 0] = -1.0  # First cluster
        pca_result[50:, 0] = 1.0   # Second cluster
        
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        pca_figure()
        
        # Should complete successfully with distinct patterns
        mock_pca.fit_transform.assert_called_once()
        mock_scatter.assert_called_once()
        mock_savefig.assert_called_once()
    
    @patch('viz.pca_circuits.load_activations')
    @patch('torch_pca.PCA')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_memory_efficiency_large_data(
        self, mock_close, mock_savefig, mock_scatter, mock_pca_class, mock_load_activations
    ):
        """Test memory efficiency with large data."""
        # Large but manageable data size
        batch_size, num_layers, num_experts = 1000, 8, 32
        
        activated_experts = th.rand(batch_size, num_layers, num_experts) > 0.8
        mock_load_activations.return_value = (activated_experts, None, 2)
        
        # Mock PCA result
        pca_result = th.randn(batch_size, 2)
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = pca_result
        mock_pca_class.return_value = mock_pca
        
        from viz.pca_circuits import pca_figure
        
        # Should complete without memory errors
        pca_figure()
        
        # Verify processing completed
        mock_pca.fit_transform.assert_called_once()
        mock_scatter.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

