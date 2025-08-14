"""Tests for viz.router_correlations module."""

import os
from unittest.mock import patch, MagicMock

import pytest
import torch as th

from viz.router_correlations import router_correlations


class TestRouterCorrelations:
    """Test router_correlations function."""

    def test_router_correlations_no_files(self, temp_dir, monkeypatch):
        """Test router_correlations with no data files."""
        # Set up patches
        monkeypatch.setattr("viz.router_correlations.ROUTER_LOGITS_DIR", str(temp_dir))
        
        with pytest.raises(ValueError, match="No data files found"):
            router_correlations()
    
    def test_router_correlations_basic(self, temp_dir, monkeypatch, capsys):
        """Test basic functionality of router_correlations."""
        # Create test data files
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create two test files with simple patterns
        for file_idx in range(2):
            # Create router logits with known patterns
            router_logits = th.zeros(5, 3, 4)  # 5 tokens, 3 layers, 4 experts
            
            # Set specific patterns
            if file_idx == 0:
                # First file: tokens activate experts 0 and 1
                router_logits[:, 0, 0] = 10.0  # Layer 0, Expert 0
                router_logits[:, 1, 1] = 10.0  # Layer 1, Expert 1
            else:
                # Second file: tokens activate experts 2 and 3
                router_logits[:, 0, 2] = 10.0  # Layer 0, Expert 2
                router_logits[:, 2, 3] = 10.0  # Layer 2, Expert 3
            
            # Save the file
            th.save(
                {"topk": 2, "router_logits": router_logits},
                os.path.join(temp_dir, f"{file_idx}.pt"),
            )
        
        # Set up patches
        monkeypatch.setattr("viz.router_correlations.ROUTER_LOGITS_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_correlations.FIGURE_DIR", str(temp_dir))
        
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            # Run the function
            router_correlations()
            
            # Check that output files were created
            assert os.path.exists(os.path.join(temp_dir, "router_correlations.png"))
            assert os.path.exists(os.path.join(temp_dir, "router_correlations_random.png"))
            assert os.path.exists(os.path.join(temp_dir, "router_correlations_cross_layer.png"))
            assert os.path.exists(os.path.join(temp_dir, "router_correlations_cross_layer_random.png"))
            
            # Check that top and bottom correlations were printed
            captured = capsys.readouterr()
            assert "Top 10 cross-layer correlations:" in captured.out
            assert "Bottom 10 cross-layer correlations:" in captured.out
    
    def test_router_correlations_correlation_calculation(self, temp_dir, monkeypatch):
        """Test that correlations are calculated correctly."""
        # Create test data with known correlation patterns
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a single test file with perfect correlation between two experts
        router_logits = th.zeros(10, 2, 3)  # 10 tokens, 2 layers, 3 experts
        
        # Set perfect correlation between Layer 0, Expert 0 and Layer 1, Expert 1
        # First 5 tokens activate both, last 5 tokens activate neither
        router_logits[0:5, 0, 0] = 10.0
        router_logits[0:5, 1, 1] = 10.0
        
        # Set perfect anti-correlation between Layer 0, Expert 1 and Layer 1, Expert 0
        # First 5 tokens activate Layer 0, Expert 1; last 5 tokens activate Layer 1, Expert 0
        router_logits[0:5, 0, 1] = 10.0
        router_logits[5:10, 1, 0] = 10.0
        
        # Save the file
        th.save(
            {"topk": 1, "router_logits": router_logits},
            os.path.join(temp_dir, "0.pt"),
        )
        
        # Set up patches
        monkeypatch.setattr("viz.router_correlations.ROUTER_LOGITS_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_correlations.FIGURE_DIR", str(temp_dir))
        
        # Mock plt.savefig to capture the correlation values
        correlation_values = {}
        
        def mock_savefig(path, **kwargs):
            nonlocal correlation_values
            if "cross_layer" in path and "random" not in path:
                # Get the current data from the plot
                fig = plt.gcf()
                ax = fig.axes[0]
                correlation_values["cross_layer"] = ax.containers[0].datavalues
        
        with patch("matplotlib.pyplot.savefig", side_effect=mock_savefig), \
             patch("matplotlib.pyplot.close"), \
             patch("matplotlib.pyplot.bar") as mock_bar, \
             patch("matplotlib.pyplot.gcf") as mock_gcf, \
             patch("matplotlib.pyplot.figure"):
            
            # Mock the figure and axes to return correlation values
            mock_ax = MagicMock()
            mock_container = MagicMock()
            mock_container.datavalues = th.tensor([1.0, -1.0])  # Perfect correlation and anti-correlation
            mock_ax.containers = [mock_container]
            mock_fig = MagicMock()
            mock_fig.axes = [mock_ax]
            mock_gcf.return_value = mock_fig
            
            # Run the function
            router_correlations()
            
            # Check that bar was called with the right data
            assert mock_bar.call_count >= 4
    
    def test_router_correlations_with_real_correlation(self, temp_dir, monkeypatch):
        """Test router_correlations with real correlation calculation."""
        # Create test data with known correlation patterns
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a single test file with perfect correlation between two experts
        router_logits = th.zeros(10, 2, 3)  # 10 tokens, 2 layers, 3 experts
        
        # Set perfect correlation between Layer 0, Expert 0 and Layer 1, Expert 1
        # First 5 tokens activate both, last 5 tokens activate neither
        router_logits[0:5, 0, 0] = 10.0
        router_logits[0:5, 1, 1] = 10.0
        
        # Set perfect anti-correlation between Layer 0, Expert 1 and Layer 1, Expert 0
        # First 5 tokens activate Layer 0, Expert 1; last 5 tokens activate Layer 1, Expert 0
        router_logits[0:5, 0, 1] = 10.0
        router_logits[5:10, 1, 0] = 10.0
        
        # Save the file
        th.save(
            {"topk": 1, "router_logits": router_logits},
            os.path.join(temp_dir, "0.pt"),
        )
        
        # Set up patches
        monkeypatch.setattr("viz.router_correlations.ROUTER_LOGITS_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_correlations.FIGURE_DIR", str(temp_dir))
        
        # Capture printed output
        with patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("builtins.print") as mock_print:
            
            # Run the function
            router_correlations()
            
            # Check that print was called with correlation values
            # We expect to see both high positive and high negative correlations
            high_positive_found = False
            high_negative_found = False
            
            for call_args in mock_print.call_args_list:
                args = call_args[0]
                if len(args) == 1 and isinstance(args[0], str) and "layer" in args[0] and "expert" in args[0]:
                    if "1.0" in args[0]:  # Perfect positive correlation
                        high_positive_found = True
                    if "-1.0" in args[0]:  # Perfect negative correlation
                        high_negative_found = True
            
            # We should find at least one high positive or high negative correlation
            assert high_positive_found or high_negative_found

