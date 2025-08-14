"""Tests for exp.kmeans_circuits module."""

import os
from unittest.mock import patch

import pytest
import torch as th

from exp.kmeans_circuits import (
    cluster_circuits,
    elbow,
    get_top_circuits,
    kmeans_manhattan,
)


class TestKmeansManhattan:
    """Test kmeans_manhattan function."""

    def test_kmeans_manhattan_basic(self):
        """Test basic functionality of kmeans_manhattan."""
        # Create a simple dataset with clear clusters
        data = th.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.9, 0.9],
                [1.0, 1.0],
            ]
        )
        k = 2

        # Run kmeans
        centroids = kmeans_manhattan(data, k, max_iters=100, seed=42)

        # Check output shape
        assert centroids.shape == (k, 2)

        # Check that centroids are close to expected values
        # Should find centroids near [0.05, 0.05] and [0.95, 0.95]
        assert th.allclose(
            centroids[0], th.tensor([0.05, 0.05]), atol=0.1
        ) or th.allclose(centroids[1], th.tensor([0.05, 0.05]), atol=0.1)
        assert th.allclose(
            centroids[0], th.tensor([0.95, 0.95]), atol=0.1
        ) or th.allclose(centroids[1], th.tensor([0.95, 0.95]), atol=0.1)

    def test_kmeans_manhattan_convergence(self):
        """Test that kmeans_manhattan converges."""
        # Create random data
        data = th.rand(100, 10)
        k = 5

        # Run kmeans with low max_iters to ensure it doesn't reach convergence
        centroids_low_iters = kmeans_manhattan(data, k, max_iters=1, seed=42)

        # Run kmeans with high max_iters to ensure it reaches convergence
        centroids_high_iters = kmeans_manhattan(data, k, max_iters=100, seed=42)

        # Check that centroids are different (low_iters didn't converge)
        assert not th.allclose(centroids_low_iters, centroids_high_iters)

    def test_kmeans_manhattan_seed(self):
        """Test that seed affects initialization."""
        # Create random data
        data = th.rand(100, 10)
        k = 5

        # Run kmeans with different seeds
        centroids1 = kmeans_manhattan(data, k, max_iters=1, seed=42)
        centroids2 = kmeans_manhattan(data, k, max_iters=1, seed=43)

        # Check that centroids are different due to different seeds
        assert not th.allclose(centroids1, centroids2)

    def test_kmeans_manhattan_invalid_input(self):
        """Test kmeans_manhattan with invalid input."""
        # Test with 3D tensor
        data = th.rand(10, 5, 3)
        k = 2

        with pytest.raises(AssertionError, match="Data must be of dimensions"):
            kmeans_manhattan(data, k)


class TestElbow:
    """Test elbow function."""

    def test_elbow_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of elbow."""
        # Create a simple dataset
        data = th.rand(100, 10)

        # Mock kmeans_manhattan to return fixed centroids
        def mock_kmeans(data, k, max_iters=1000, seed=0):
            return th.rand(k, data.shape[1])

        # Set up patches
        monkeypatch.setattr("exp.kmeans_circuits.FIGURE_DIR", str(temp_dir))

        with (
            patch("exp.kmeans_circuits.kmeans_manhattan", side_effect=mock_kmeans),
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.close"),
        ):
            # Run elbow method with small range for testing
            elbow(data, start=2, stop=10, step=2, seed=42)

            # Check that savefig was called
            mock_savefig.assert_called_once()
            args, _ = mock_savefig.call_args
            assert os.path.join(temp_dir, "elbow_method.png") in args[0]

    def test_elbow_invalid_input(self):
        """Test elbow with invalid input."""
        # Test with 3D tensor
        data = th.rand(10, 5, 3)

        with pytest.raises(AssertionError, match="Data must be of dimensions"):
            elbow(data)


class TestGetTopCircuits:
    """Test get_top_circuits function."""

    def test_get_top_circuits_basic(self):
        """Test basic functionality of get_top_circuits."""
        # Create test centroids
        num_centroids = 3
        num_layers = 2
        num_experts = 4
        centroids = th.rand(num_centroids, num_layers * num_experts)

        # Get top circuits
        top_k = 2
        indices, mask = get_top_circuits(centroids, num_layers, top_k)

        # Check output shapes
        assert indices.shape == (num_centroids, num_layers, top_k)
        assert mask.shape == (num_centroids, num_layers, num_experts)

        # Check that mask has exactly top_k True values per layer per centroid
        assert th.all(mask.sum(dim=2) == top_k)

    def test_get_top_circuits_values(self):
        """Test that get_top_circuits selects the correct experts."""
        # Create centroids with known values
        centroids = th.tensor(
            [
                [0.1, 0.5, 0.2, 0.3, 0.7, 0.4],  # Centroid 0
                [0.9, 0.1, 0.8, 0.2, 0.3, 0.7],  # Centroid 1
            ]
        )
        num_layers = 2
        top_k = 1

        # Expected top experts:
        # Centroid 0, Layer 0: Expert 1 (value 0.5)
        # Centroid 0, Layer 1: Expert 0 (value 0.7)
        # Centroid 1, Layer 0: Expert 0 (value 0.9)
        # Centroid 1, Layer 1: Expert 2 (value 0.7)

        # Call the function
        indices, mask = get_top_circuits(centroids, num_layers, top_k)

        # Check indices shape
        assert indices.shape == (2, 2, 1)

        # Check mask shape
        assert mask.shape == (2, 2, 3)

        # Check that the correct experts are selected
        # Centroid 0, Layer 0: Expert 1
        assert mask[0, 0, 1].item() == 1
        # Centroid 0, Layer 1: Expert 0
        assert mask[0, 1, 0].item() == 1
        # Centroid 1, Layer 0: Expert 0
        assert mask[1, 0, 0].item() == 1
        # Centroid 1, Layer 1: Expert 2
        assert mask[1, 1, 2].item() == 1

        # Check that other positions are 0
        assert mask[0, 0, 0].item() == 0
        assert mask[0, 0, 2].item() == 0
        assert mask[0, 1, 1].item() == 0
        assert mask[0, 1, 2].item() == 0
        assert mask[1, 0, 1].item() == 0
        assert mask[1, 0, 2].item() == 0
        assert mask[1, 1, 0].item() == 0
        assert mask[1, 1, 1].item() == 0


class TestClusterCircuits:
    """Test cluster_circuits function."""

    def test_cluster_circuits_with_k(self, temp_dir, monkeypatch):
        """Test cluster_circuits with specified k."""
        # Mock dependencies
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)
        mock_centroids = th.rand(5, 12)  # 5 centroids, 12 features (3*4)

        # Set up patches
        monkeypatch.setattr("exp.kmeans_circuits.OUTPUT_DIR", str(temp_dir))

        with (
            patch(
                "exp.kmeans_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ),
            patch(
                "exp.kmeans_circuits.kmeans_manhattan",
                return_value=mock_centroids,
            ),
            patch(
                "torch.cuda.is_available",
                return_value=False,
            ),
        ):
            # Run the function with k=5
            cluster_circuits(k=5, seed=42)

            # Check that output file was created
            output_file = os.path.join(temp_dir, "kmeans_circuits.pt")
            assert os.path.exists(output_file)

            # Load and verify the output
            saved_data = th.load(output_file)
            assert "circuits" in saved_data
            assert "top_k" in saved_data
            assert th.equal(saved_data["circuits"], mock_centroids)
            assert saved_data["top_k"] == 2

    def test_cluster_circuits_without_k(self, monkeypatch):
        """Test cluster_circuits without k (should run elbow method)."""
        # Mock dependencies
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)

        with (
            patch(
                "exp.kmeans_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ),
            patch(
                "exp.kmeans_circuits.elbow",
            ) as mock_elbow,
            patch(
                "torch.cuda.is_available",
                return_value=False,
            ),
        ):
            # Run the function without k
            cluster_circuits(k=None, seed=42)

            # Check that elbow was called
            mock_elbow.assert_called_once()
            args, kwargs = mock_elbow.call_args
            assert th.equal(args[0], mock_activated_experts.view(10, -1).float())
            assert kwargs["seed"] == 42
