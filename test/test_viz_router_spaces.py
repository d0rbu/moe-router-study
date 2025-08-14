"""Tests for viz.router_spaces module."""

import os
from unittest.mock import patch, MagicMock

import pytest
import torch as th

from viz.router_spaces import router_spaces, ROUTER_VIZ_DIR


class TestRouterSpaces:
    """Test router_spaces function."""

    def test_router_spaces_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of router_spaces."""
        # Create mock weight data
        mock_router_weights = {
            0: th.randn(4, 16),  # 4 experts, 16-dim hidden
            1: th.randn(4, 16),
        }
        mock_down_proj_weights = {
            0: th.randn(4, 16, 64),  # 4 experts, 16-dim hidden, 64-dim MLP
            1: th.randn(4, 16, 64),
        }
        mock_o_proj_weights = {
            0: th.randn(16, 16),  # 16-dim hidden
            1: th.randn(16, 16),
        }
        
        # Create mock data files
        mock_router_data = {
            "weights": mock_router_weights,
            "topk": 2,
        }
        mock_down_proj_data = {
            "weights": mock_down_proj_weights,
        }
        mock_o_proj_data = {
            "weights": mock_o_proj_weights,
        }
        
        # Set up patches
        monkeypatch.setattr("viz.router_spaces.WEIGHT_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.FIGURE_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.ROUTER_VIZ_DIR", os.path.join(temp_dir, "router_spaces"))
        
        with patch(
            "torch.load",
            side_effect=[mock_router_data, mock_down_proj_data, mock_o_proj_data],
        ), patch(
            "matplotlib.pyplot.plot",
        ), patch(
            "matplotlib.pyplot.savefig",
        ), patch(
            "matplotlib.pyplot.close",
        ), patch(
            "torch.linalg.svdvals",
            return_value=th.tensor([1.0, 0.5, 0.2, 0.1]),
        ), patch(
            "torch.linalg.svd",
            return_value=(
                th.randn(8, 8),  # u
                th.tensor([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]),  # s
                th.randn(8, 32),  # vh
            ),
        ):
            # Run the function
            router_spaces()
            
            # Check that the output directory was created
            assert os.path.exists(os.path.join(temp_dir, "router_spaces"))
            
            # Check that plot was called multiple times
            plt_plot = pytest.importorskip("matplotlib.pyplot").plot
            assert plt_plot.call_count > 0
            
            # Check that savefig was called multiple times
            plt_savefig = pytest.importorskip("matplotlib.pyplot").savefig
            assert plt_savefig.call_count > 0
    
    def test_router_spaces_with_small_svd(self, temp_dir, monkeypatch):
        """Test router_spaces with small SVD result (fewer than 100 singular vectors)."""
        # Create mock weight data with small dimensions
        mock_router_weights = {
            0: th.randn(2, 4),  # 2 experts, 4-dim hidden
        }
        mock_down_proj_weights = {
            0: th.randn(2, 4, 8),  # 2 experts, 4-dim hidden, 8-dim MLP
        }
        mock_o_proj_weights = {
            0: th.randn(4, 4),  # 4-dim hidden
        }
        
        # Create mock data files
        mock_router_data = {
            "weights": mock_router_weights,
            "topk": 1,
        }
        mock_down_proj_data = {
            "weights": mock_down_proj_weights,
        }
        mock_o_proj_data = {
            "weights": mock_o_proj_weights,
        }
        
        # Set up patches
        monkeypatch.setattr("viz.router_spaces.WEIGHT_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.FIGURE_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.ROUTER_VIZ_DIR", os.path.join(temp_dir, "router_spaces"))
        
        # Create a small SVD result
        u = th.randn(2, 2)  # Only 2 singular vectors
        s = th.tensor([1.0, 0.5])
        vh = th.randn(2, 2)
        
        with patch(
            "torch.load",
            side_effect=[mock_router_data, mock_down_proj_data, mock_o_proj_data],
        ), patch(
            "matplotlib.pyplot.plot",
        ), patch(
            "matplotlib.pyplot.savefig",
        ), patch(
            "matplotlib.pyplot.close",
        ), patch(
            "torch.linalg.svdvals",
            return_value=th.tensor([1.0, 0.5]),
        ), patch(
            "torch.linalg.svd",
            return_value=(u, s, vh),
        ):
            # Run the function
            router_spaces()
            
            # Check that the function completed without errors
            assert True
    
    def test_router_spaces_with_multiple_layers(self, temp_dir, monkeypatch):
        """Test router_spaces with multiple layers."""
        # Create mock weight data with multiple layers
        mock_router_weights = {
            0: th.randn(4, 16),
            1: th.randn(4, 16),
            2: th.randn(4, 16),
        }
        mock_down_proj_weights = {
            0: th.randn(4, 16, 64),
            1: th.randn(4, 16, 64),
            2: th.randn(4, 16, 64),
        }
        mock_o_proj_weights = {
            0: th.randn(16, 16),
            1: th.randn(16, 16),
            2: th.randn(16, 16),
        }
        
        # Create mock data files
        mock_router_data = {
            "weights": mock_router_weights,
            "topk": 2,
        }
        mock_down_proj_data = {
            "weights": mock_down_proj_weights,
        }
        mock_o_proj_data = {
            "weights": mock_o_proj_weights,
        }
        
        # Set up patches
        monkeypatch.setattr("viz.router_spaces.WEIGHT_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.FIGURE_DIR", str(temp_dir))
        monkeypatch.setattr("viz.router_spaces.ROUTER_VIZ_DIR", os.path.join(temp_dir, "router_spaces"))
        
        with patch(
            "torch.load",
            side_effect=[mock_router_data, mock_down_proj_data, mock_o_proj_data],
        ), patch(
            "matplotlib.pyplot.plot",
        ), patch(
            "matplotlib.pyplot.savefig",
        ), patch(
            "matplotlib.pyplot.close",
        ), patch(
            "torch.linalg.svdvals",
            return_value=th.tensor([1.0, 0.5, 0.2, 0.1]),
        ), patch(
            "torch.linalg.svd",
            return_value=(
                th.randn(12, 12),  # u
                th.tensor([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),  # s
                th.randn(12, 48),  # vh
            ),
        ):
            # Run the function
            router_spaces()
            
            # Check that the function completed without errors
            assert True

