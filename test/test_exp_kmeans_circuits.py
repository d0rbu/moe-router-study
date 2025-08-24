"""Tests for kmeans_circuits.py."""

import tempfile
from unittest.mock import patch

import pytest
import torch as th

from exp.kmeans_circuits import elbow, kmeans_manhattan


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
        centroids, assignments, losses = kmeans_manhattan(
            data, k, minibatch_size=4, max_iters=100, seed=42
        )

        # Check output shape
        assert centroids.shape == (k, 2)
        assert assignments.shape == (4,)
        assert len(losses) > 0

        # Check that assignments are correct (points should be assigned to nearest centroid)
        # First two points should be in one cluster, last two in another
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[0] != assignments[2]

    def test_kmeans_manhattan_invalid_batch_size(self):
        """Test kmeans_manhattan with invalid batch size."""
        # Create random data
        data = th.rand(100, 10)
        k = 5

        # Test with negative batch size
        with pytest.raises(
            AssertionError, match="Batch size must be > 0 and <= dataset_size"
        ):
            kmeans_manhattan(data, k, minibatch_size=-1)

        # Test with batch size greater than dataset size
        with pytest.raises(
            AssertionError, match="Batch size must be > 0 and <= dataset_size"
        ):
            kmeans_manhattan(data, k, minibatch_size=101)


class TestElbow:
    """Test elbow function."""

    def test_elbow_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of elbow."""
        # Create a simple dataset
        data = th.rand(100, 10)

        # Mock kmeans_manhattan to return fixed centroids
        def mock_kmeans(data, k, minibatch_size=None, max_iters=100, seed=0):
            centroids = th.rand(k, data.shape[1])
            assignments = th.randint(0, k, (data.shape[0],))
            losses = [float(k)]  # Make loss proportional to k for predictable elbow
            return centroids, assignments, losses

        # Patch kmeans_manhattan and plt functions
        monkeypatch.setattr("exp.kmeans_circuits.kmeans_manhattan", mock_kmeans)

        # Skip the actual plotting
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.show"):
            # Run elbow method
            elbow(data, minibatch_size=10, start=1, stop=6, step=1, seed=42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname
