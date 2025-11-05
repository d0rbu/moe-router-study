"""Tests for K-means centroid computation using scatter_add_."""

import pytest
import torch as th

from exp.kmeans import compute_all_centroids_from_assignments


class TestComputeAllCentroidsFromAssignments:
    """Test suite for the vectorized centroid computation function."""

    def test_basic_computation(self):
        """Test basic centroid computation with simple data."""
        # Create simple test data: 6 points, 2 centroids, 2D space
        data = th.tensor(
            [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [5.0, 5.0], [5.0, 6.0], [6.0, 5.0]],
            dtype=th.float32,
        )
        assignments = th.tensor([0, 0, 0, 1, 1, 1], dtype=th.int64)
        num_centroids = 2

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (2, 2)
        assert weights.shape == (2,)

        # Check weights
        assert weights[0] == 3
        assert weights[1] == 3
        assert weights.sum() == 6

        # Check centroid values
        expected_centroid_0 = th.tensor([4.0 / 3.0, 4.0 / 3.0], dtype=th.float32)
        expected_centroid_1 = th.tensor([16.0 / 3.0, 16.0 / 3.0], dtype=th.float32)

        assert th.allclose(new_centroids[0], expected_centroid_0, atol=1e-5)
        assert th.allclose(new_centroids[1], expected_centroid_1, atol=1e-5)

    def test_empty_clusters(self):
        """Test handling of empty clusters."""
        # 4 points assigned to only 2 of 3 centroids
        data = th.tensor(
            [[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]], dtype=th.float32
        )
        assignments = th.tensor([0, 0, 2, 2], dtype=th.int64)
        num_centroids = 3

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (3, 2)
        assert weights.shape == (3,)

        # Check weights
        assert weights[0] == 2
        assert weights[1] == 0  # Empty cluster
        assert weights[2] == 2
        assert weights.sum() == 4

        # Check that empty cluster has zero centroid
        assert th.allclose(new_centroids[1], th.zeros(2), atol=1e-5)

        # Check non-empty centroids
        expected_centroid_0 = th.tensor([1.0, 1.5], dtype=th.float32)
        expected_centroid_2 = th.tensor([5.0, 5.5], dtype=th.float32)

        assert th.allclose(new_centroids[0], expected_centroid_0, atol=1e-5)
        assert th.allclose(new_centroids[2], expected_centroid_2, atol=1e-5)

    def test_single_point_per_cluster(self):
        """Test when each cluster has exactly one point."""
        data = th.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=th.float32)
        assignments = th.tensor([0, 1, 2], dtype=th.int64)
        num_centroids = 3

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (3, 2)
        assert weights.shape == (3,)

        # Each cluster should have weight 1
        assert th.all(weights == 1)
        assert weights.sum() == 3

        # Centroids should equal the original points
        assert th.allclose(new_centroids, data, atol=1e-5)

    def test_all_points_same_cluster(self):
        """Test when all points are assigned to the same cluster."""
        data = th.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=th.float32
        )
        assignments = th.tensor([1, 1, 1, 1], dtype=th.int64)
        num_centroids = 3

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (3, 2)
        assert weights.shape == (3,)

        # Check weights
        assert weights[0] == 0
        assert weights[1] == 4
        assert weights[2] == 0
        assert weights.sum() == 4

        # Cluster 1 should have the mean of all points
        expected_centroid = th.tensor([2.5, 2.5], dtype=th.float32)
        assert th.allclose(new_centroids[1], expected_centroid, atol=1e-5)

        # Other clusters should be zero
        assert th.allclose(new_centroids[0], th.zeros(2), atol=1e-5)
        assert th.allclose(new_centroids[2], th.zeros(2), atol=1e-5)

    def test_higher_dimensions(self):
        """Test with higher dimensional data."""
        # 8 points, 2 clusters, 5D space
        data = th.randn(8, 5, dtype=th.float32)
        assignments = th.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=th.int64)
        num_centroids = 2

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (2, 5)
        assert weights.shape == (2,)

        # Check weights
        assert weights[0] == 4
        assert weights[1] == 4
        assert weights.sum() == 8

        # Manually compute expected centroids
        expected_centroid_0 = data[0:4].mean(dim=0)
        expected_centroid_1 = data[4:8].mean(dim=0)

        assert th.allclose(new_centroids[0], expected_centroid_0, atol=1e-5)
        assert th.allclose(new_centroids[1], expected_centroid_1, atol=1e-5)

    def test_large_scale(self):
        """Test with realistic large-scale data."""
        # Simulate a batch of 4000 points with 1024 centroids in 512D space
        batch_size = 4000
        embed_dim = 512
        num_centroids = 1024

        data = th.randn(batch_size, embed_dim, dtype=th.float32)
        assignments = th.randint(0, num_centroids, (batch_size,), dtype=th.int64)

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, num_centroids
        )

        # Check shapes
        assert new_centroids.shape == (num_centroids, embed_dim)
        assert weights.shape == (num_centroids,)

        # Check that weights sum to batch size
        assert weights.sum() == batch_size

        # Check that all weights are non-negative
        assert th.all(weights >= 0)

        # Check no NaN values
        assert not th.isnan(new_centroids).any()
        assert not th.isnan(weights).any()

    def test_dtype_preservation(self):
        """Test that data types are preserved correctly."""
        # Test with float64
        data_f64 = th.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=th.float64)
        assignments = th.tensor([0, 1], dtype=th.int64)

        centroids_f64, weights = compute_all_centroids_from_assignments(
            data_f64, assignments, 2
        )

        assert centroids_f64.dtype == th.float64
        assert weights.dtype == th.int64

        # Test with float16
        data_f16 = th.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=th.float16)
        centroids_f16, weights = compute_all_centroids_from_assignments(
            data_f16, assignments, 2
        )

        assert centroids_f16.dtype == th.float16
        assert weights.dtype == th.int64

    @pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test computation on CUDA device."""
        data = th.tensor(
            [[1.0, 1.0], [1.0, 2.0], [5.0, 5.0], [5.0, 6.0]], dtype=th.float32
        ).cuda()
        assignments = th.tensor([0, 0, 1, 1], dtype=th.int64).cuda()

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, 2
        )

        # Check that outputs are on CUDA
        assert new_centroids.is_cuda
        assert weights.is_cuda

        # Check correctness
        expected_centroid_0 = th.tensor([1.0, 1.5], dtype=th.float32)
        expected_centroid_1 = th.tensor([5.0, 5.5], dtype=th.float32)

        assert th.allclose(new_centroids[0].cpu(), expected_centroid_0, atol=1e-5)
        assert th.allclose(new_centroids[1].cpu(), expected_centroid_1, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values
        data_large = th.tensor([[1e6, 1e6], [1e6, 1e6 + 1]], dtype=th.float32)
        assignments = th.tensor([0, 0], dtype=th.int64)

        centroids_large, _ = compute_all_centroids_from_assignments(
            data_large, assignments, 1
        )

        expected = th.tensor([1e6, 1e6 + 0.5], dtype=th.float32)
        assert th.allclose(centroids_large[0], expected, rtol=1e-4)

        # Test with very small values
        data_small = th.tensor([[1e-6, 1e-6], [2e-6, 2e-6]], dtype=th.float32)
        centroids_small, _ = compute_all_centroids_from_assignments(
            data_small, assignments, 1
        )

        expected_small = th.tensor([1.5e-6, 1.5e-6], dtype=th.float32)
        assert th.allclose(centroids_small[0], expected_small, rtol=1e-4)

    def test_scatter_add_correctness(self):
        """Test that scatter_add_ produces correct sums."""
        # Create data where we can manually verify scatter_add results
        data = th.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=th.float32,
        )
        # Assign: [0, 0, 1, 1]
        assignments = th.tensor([0, 0, 1, 1], dtype=th.int64)

        new_centroids, weights = compute_all_centroids_from_assignments(
            data, assignments, 2
        )

        # Cluster 0: mean of [[1, 10], [2, 20]] = [1.5, 15.0]
        # Cluster 1: mean of [[3, 30], [4, 40]] = [3.5, 35.0]
        expected = th.tensor([[1.5, 15.0], [3.5, 35.0]], dtype=th.float32)

        assert th.allclose(new_centroids, expected, atol=1e-5)
        assert weights[0] == 2
        assert weights[1] == 2
