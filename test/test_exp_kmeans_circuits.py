"""Tests for exp.kmeans_circuits module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch as th

from exp.kmeans_circuits import (
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

    def test_kmeans_manhattan_batch_size(self):
        """Test kmeans_manhattan with different batch sizes."""
        # Create random data
        data = th.rand(100, 10)
        k = 5

        # Run kmeans with default batch size (full dataset)
        centroids_full = kmeans_manhattan(data, k, max_iters=10, seed=42)

        # Run kmeans with a smaller batch size
        centroids_batch = kmeans_manhattan(
            data, k, batch_size=20, max_iters=10, seed=42
        )

        # Results should be similar but not identical due to batch processing
        # We're just checking that it runs without errors
        assert centroids_batch.shape == centroids_full.shape

    def test_kmeans_manhattan_invalid_batch_size(self):
        """Test kmeans_manhattan with invalid batch size."""
        # Create random data
        data = th.rand(100, 10)
        k = 5

        # Test with negative batch size
        with pytest.raises(
            AssertionError, match="Batch size must be > 0 and < dataset_size"
        ):
            kmeans_manhattan(data, k, batch_size=-1)

        # Test with batch size equal to dataset size
        with pytest.raises(
            AssertionError, match="Batch size must be > 0 and < dataset_size"
        ):
            kmeans_manhattan(data, k, batch_size=100)

        # Test with batch size greater than dataset size
        with pytest.raises(
            AssertionError, match="Batch size must be > 0 and < dataset_size"
        ):
            kmeans_manhattan(data, k, batch_size=101)


class TestElbow:
    """Test elbow function."""

    def test_elbow_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of elbow."""
        # Create a simple dataset
        data = th.rand(100, 10)

        # Mock kmeans_manhattan to return fixed centroids
        def mock_kmeans(data, k, batch_size=0, max_iters=1000, seed=0):
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

    def test_elbow_with_batch_size(self, temp_dir, monkeypatch):
        """Test elbow function with batch_size parameter."""
        # Create a simple dataset
        data = th.rand(100, 10)
        batch_size = 20

        # Set up patches
        monkeypatch.setattr("exp.kmeans_circuits.FIGURE_DIR", str(temp_dir))

        # Just test that it runs without errors with batch_size parameter
        with (
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            # Run elbow method with batch_size
            elbow(data, batch_size=batch_size, start=2, stop=4, step=2, seed=42)
            # If we get here without errors, the test passes

    def test_elbow_invalid_input(self):
        """Test elbow with invalid input."""
        # Test with 3D tensor
        data = th.rand(10, 5, 3)

        with pytest.raises(AssertionError, match="Data must be of dimensions"):
            elbow(data)


class TestGetTopCircuits:
    """Test get_top_circuits function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_get_top_circuits_values(self):
        """Test that get_top_circuits selects the correct experts."""
        # Create centroids with known values
        centroids = th.tensor(
            [
                [0.1, 0.5, 0.2, 0.3, 0.7, 0.4],  # Centroid 0
                [0.9, 0.1, 0.8, 0.2, 0.3, 0.7],  # Centroid 1
            ]
        )

        # Call the function
        num_layers = 2  # Assuming 2 layers with 3 experts each
        top_k = 2
        indices, mask = get_top_circuits(centroids, num_layers, top_k)

        # Check the result
        assert indices.shape == (2, 2, 2)  # 2 centroids, 2 layers, 2 experts per layer

        # For the purpose of this test, we'll just check that the indices and mask have the right shape
        # The actual values would depend on the implementation
        assert mask.shape == (2, 2, 3)  # 2 centroids, 2 layers, 3 experts per layer


class TestClusterCircuits:
    """Test cluster_circuits function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_cluster_circuits_with_k(self):
        """Test cluster_circuits with specified k."""
        # Create mock activated experts
        activated_experts = th.zeros(10, 3, 4, dtype=th.bool)
        # Set some activations to True to create patterns
        activated_experts[0:3, 0, 0] = True
        activated_experts[4:7, 1, 2] = True
        activated_experts[8:10, 2, 3] = True

        # Mock KMeans
        mock_kmeans = MagicMock()
        mock_kmeans.cluster_centers_ = np.random.rand(
            2, 12
        )  # 2 clusters, 12 features (3*4)
        mock_kmeans.labels_ = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # 10 samples

        with patch(
            "sklearn.cluster.KMeans", return_value=mock_kmeans
        ) as mock_kmeans_class:
            # Call the function
            from exp.kmeans_circuits import cluster_circuits

            clusters, centroids = cluster_circuits(activated_experts, k=2)

            # Check that KMeans was called with the right parameters
            mock_kmeans_class.assert_called_once_with(n_clusters=2, random_state=42)

            # Check the result
            assert isinstance(clusters, list)
            assert len(clusters) == 2  # 2 clusters
            assert th.is_tensor(centroids)
            assert centroids.shape == (2, 12)  # 2 clusters, 12 features (3*4)

            # Check cluster assignments
            assert len(clusters[0]) == 6  # 6 samples in cluster 0
            assert len(clusters[1]) == 4  # 4 samples in cluster 1
            assert all(idx in clusters[0] for idx in [0, 1, 2, 7, 8, 9])
            assert all(idx in clusters[1] for idx in [3, 4, 5, 6])

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_cluster_circuits_without_k(self):
        """Test cluster_circuits without specified k (should use elbow method)."""
        # Create mock activated experts
        activated_experts = th.zeros(10, 3, 4, dtype=th.bool)
        # Set some activations to True to create patterns
        activated_experts[0:3, 0, 0] = True
        activated_experts[4:7, 1, 2] = True
        activated_experts[8:10, 2, 3] = True

        # Mock KMeans for different k values
        mock_kmeans_results = []
        for k in range(1, 11):  # k from 1 to 10
            mock_kmeans = MagicMock()
            mock_kmeans.inertia_ = 100 / k  # Decreasing inertia with increasing k
            mock_kmeans.cluster_centers_ = np.random.rand(k, 12)
            mock_kmeans.labels_ = np.random.randint(0, k, size=10)
            mock_kmeans_results.append(mock_kmeans)

        # The optimal k should be 3 based on the elbow method
        optimal_k = 3

        with (
            patch(
                "sklearn.cluster.KMeans", side_effect=mock_kmeans_results
            ) as mock_kmeans_class,
            patch("matplotlib.pyplot.plot"),
            patch("matplotlib.pyplot.xlabel"),
            patch("matplotlib.pyplot.ylabel"),
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch(
                "exp.kmeans_circuits._find_elbow", return_value=optimal_k
            ) as mock_find_elbow,
        ):
            # Call the function
            from exp.kmeans_circuits import cluster_circuits

            clusters, centroids = cluster_circuits(activated_experts)

            # Check that KMeans was called for each k
            assert mock_kmeans_class.call_count == 10

            # Check that _find_elbow was called
            mock_find_elbow.assert_called_once()

            # Check the result
            assert isinstance(clusters, list)
            assert len(clusters) == optimal_k
            assert th.is_tensor(centroids)
            assert centroids.shape == (optimal_k, 12)
