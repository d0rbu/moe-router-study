"""Tests for viz.dbscan_circuits module."""

from unittest.mock import MagicMock, patch

import torch as th

from viz.dbscan_circuits import cluster_circuits


class TestClusterCircuits:
    """Test cluster_circuits function."""

    def test_cluster_circuits_basic(self, capsys):
        """Test basic functionality of cluster_circuits."""
        # Mock load_activations_and_topk to return a known tensor
        mock_activations = th.zeros(10, 3, 4, dtype=th.bool)
        # Set some activations to True to create patterns
        mock_activations[0:3, 0, 0] = (
            True  # First 3 samples activate expert 0 in layer 0
        )
        mock_activations[4:7, 1, 2] = True  # Samples 4-6 activate expert 2 in layer 1
        mock_activations[8:10, 2, 3] = (
            True  # Last 2 samples activate expert 3 in layer 2
        )

        # Mock DBSCAN result
        mock_clusters = th.tensor([0, 0, 0, -1, 1, 1, 1, -1, 2, 2])

        with (
            patch(
                "viz.dbscan_circuits.load_activations_and_topk",
                return_value=(mock_activations, 1),
            ),
            patch(
                "sklearn.cluster.DBSCAN.fit_predict",
                return_value=mock_clusters,
            ),
        ):
            # Run the function
            from viz.dbscan_circuits import cluster_circuits

            cluster_circuits()

            # Check that output was printed
            captured = capsys.readouterr()
            assert "tensor" in captured.out or "array" in captured.out

    def test_cluster_circuits_with_dbscan(self):
        """Test that DBSCAN is called with correct parameters."""
        # Mock load_activations_and_topk
        mock_activations = th.zeros(10, 3, 4, dtype=th.bool)

        # Mock DBSCAN to track calls
        mock_dbscan = MagicMock()
        mock_dbscan.fit_predict.return_value = th.zeros(10, dtype=th.int64)

        with (
            patch(
                "viz.dbscan_circuits.load_activations_and_topk",
                return_value=(mock_activations, 1),
            ),
            patch(
                "viz.dbscan_circuits.DBSCAN",
                return_value=mock_dbscan,
            ) as mock_dbscan_cls,
        ):
            # Run the function
            cluster_circuits()

            # Check that DBSCAN was called with correct parameters
            mock_dbscan_cls.assert_called_once_with(eps=0.1, min_samples=10)

            # Check that fit_predict was called
            mock_dbscan.fit_predict.assert_called_once()

            # Check that the input to fit_predict has the right shape
            args, _ = mock_dbscan.fit_predict.call_args
            assert args[0].shape == (10, 12)  # 10 samples, 3*4=12 flattened features

    def test_cluster_circuits_reshape(self):
        """Test that activations are reshaped correctly."""
        # Create a test tensor with known shape
        mock_activations = th.zeros(5, 2, 3, dtype=th.bool)

        with (
            patch(
                "viz.dbscan_circuits.load_activations_and_topk",
                return_value=(mock_activations, 1),
            ),
            patch(
                "viz.dbscan_circuits.DBSCAN.fit_predict",
                return_value=th.zeros(5, dtype=th.int64),
            ) as mock_fit_predict,
        ):
            # Run the function
            cluster_circuits()

            # Check that fit_predict was called with reshaped tensor
            mock_fit_predict.assert_called_once()
            args, _ = mock_fit_predict.call_args
            assert args[0].shape == (5, 6)  # 5 samples, 2*3=6 flattened features
            assert args[0].dtype == th.float32  # Should be converted to float

    def test_cluster_circuits_with_real_dbscan(self):
        """Test with actual DBSCAN implementation."""
        # Create a test tensor with clear clusters
        mock_activations = th.zeros(20, 2, 2, dtype=th.bool)

        # Create two distinct clusters
        mock_activations[0:10, 0, 0] = (
            True  # First 10 samples activate expert 0 in layer 0
        )
        mock_activations[10:20, 1, 1] = (
            True  # Last 10 samples activate expert 1 in layer 1
        )

        with (
            patch(
                "viz.dbscan_circuits.load_activations_and_topk",
                return_value=(mock_activations, 1),
            ),
            patch(
                "viz.dbscan_circuits.print",
            ) as mock_print,
        ):
            # Run the function
            cluster_circuits()

            # Check that print was called with clusters
            mock_print.assert_called_once()
            args, _ = mock_print.call_args
            clusters = args[0]

            # Should have found 2 clusters (plus possibly noise points labeled as -1)
            unique_clusters = set(clusters.tolist())
            # Remove -1 (noise) if present
            if -1 in unique_clusters:
                unique_clusters.remove(-1)

            # Should have at least 2 clusters
            assert len(unique_clusters) >= 2
