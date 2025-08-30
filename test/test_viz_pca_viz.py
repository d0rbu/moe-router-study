"""Tests for viz.pca_viz module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch as th

from viz.pca_viz import pca_figure


class TestPcaFigure:
    """Test pca_figure function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_pca_figure_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of pca_figure."""
        # Create test data
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)
        # Set some activations to True to create patterns
        mock_activated_experts[0:3, 0, 0] = True
        mock_activated_experts[4:7, 1, 2] = True
        mock_activated_experts[8:10, 2, 3] = True

        # Mock PCA result
        mock_pca_result = th.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
                [1.3, 1.4],
                [1.5, 1.6],
                [1.7, 1.8],
                [1.9, 2.0],
            ]
        )

        # Set up patches
        figure_dir = os.path.join(str(temp_dir), "test_experiment")
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, "pca_circuits.png")

        # Create mock PCA class
        mock_pca = MagicMock()
        mock_pca.fit_transform.return_value = mock_pca_result

        with (
            patch(
                "exp.activations.load_activations",
                return_value=mock_activated_experts,
            ),
            patch(
                "torch_pca.PCA",
                return_value=mock_pca,
            ),
            patch(
                "matplotlib.pyplot.scatter",
            ) as mock_scatter,
            patch(
                "matplotlib.pyplot.savefig",
            ) as mock_savefig,
            patch(
                "matplotlib.pyplot.close",
            ),
            patch(
                "viz.get_figure_dir",
                return_value=figure_dir,
            ),
        ):
            # Run the function
            pca_figure(device="cpu", experiment_name="test_experiment")

            # Check that PCA was called with the right data
            mock_pca.fit_transform.assert_called_once()
            args, _ = mock_pca.fit_transform.call_args
            assert args[0].shape == (10, 12)  # 10 samples, 12 features (3*4)

            # Check that scatter was called with the PCA result
            mock_scatter.assert_called_once()

            # Check that savefig was called with the right path
            mock_savefig.assert_called_once()
            args, _ = mock_savefig.call_args
            assert args[0] == figure_path

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_pca_figure_with_device(self):
        """Test pca_figure with specified device."""
        with (
            patch(
                "exp.activations.load_activations",
            ) as mock_load,
            patch(
                "torch_pca.PCA",
                return_value=MagicMock(),
            ),
            patch(
                "matplotlib.pyplot.scatter",
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ),
            patch(
                "matplotlib.pyplot.close",
            ),
            patch(
                "os.makedirs",
            ),
        ):
            # Run the function with device="cuda"
            pca_figure(device="cuda")

            # Check that load_activations was called with device="cuda"
            mock_load.assert_called_once_with(device="cuda")

    def test_pca_figure_creates_directory(self, tmp_path):
        """Test that pca_figure creates the figure directory if it doesn't exist."""
        # Set up figure directory
        figure_dir = os.path.join(str(tmp_path), "fig", "test_experiment")
        
        # Remove the directory if it exists
        if os.path.exists(figure_dir):
            import shutil
            shutil.rmtree(os.path.dirname(figure_dir))

        with (
            patch(
                "exp.activations.load_activations",
                return_value=th.zeros(10, 3, 4, dtype=th.bool),
            ),
            patch(
                "torch_pca.PCA",
                return_value=MagicMock(),
            ),
            patch(
                "matplotlib.pyplot.scatter",
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ),
            patch(
                "matplotlib.pyplot.close",
            ),
            patch(
                "viz.get_figure_dir",
                return_value=figure_dir,
            ),
            patch(
                "os.makedirs",
                wraps=os.makedirs,
            ) as mock_makedirs,
        ):
            # Run the function
            pca_figure(device="cpu", experiment_name="test_experiment")

            # Check that makedirs was called with the figure directory
            mock_makedirs.assert_called_with(figure_dir, exist_ok=True)

