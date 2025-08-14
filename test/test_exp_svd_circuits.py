"""Tests for exp.svd_circuits module."""

import os
from unittest.mock import patch

import torch as th

from exp.svd_circuits import svd_circuits


class TestSvdCircuits:
    """Test svd_circuits function."""

    def test_svd_circuits_basic(self, temp_dir, monkeypatch):
        """Test basic functionality of svd_circuits."""
        # Create test data
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)
        # Set some activations to True to create patterns
        mock_activated_experts[0:3, 0, 0] = True
        mock_activated_experts[4:7, 1, 2] = True
        mock_activated_experts[8:10, 2, 3] = True

        # Mock SVD result
        mock_u = th.rand(10, 10)
        mock_s = th.tensor([0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
        mock_vh = th.rand(10, 12)  # 12 = 3*4 (layers * experts)

        # Set up patches
        monkeypatch.setattr("exp.svd_circuits.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr("exp.svd_circuits.FIGURE_DIR", str(temp_dir))

        with (
            patch(
                "exp.svd_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ),
            patch(
                "torch.linalg.svd",
                return_value=(mock_u, mock_s, mock_vh),
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ) as mock_savefig,
            patch(
                "matplotlib.pyplot.close",
            ),
        ):
            # Run the function
            svd_circuits(num_circuits=5, device="cpu")

            # Check that savefig was called
            mock_savefig.assert_called_once()

            # Check that output file was created
            output_file = os.path.join(temp_dir, "svd_circuits.pt")
            assert os.path.exists(output_file)

            # Load and verify the output
            saved_data = th.load(output_file)
            assert "circuits" in saved_data
            assert "top_k" in saved_data
            assert th.equal(saved_data["circuits"], mock_vh[:5, :])
            assert saved_data["top_k"] == 2

    def test_svd_circuits_with_batch_size(self, temp_dir, monkeypatch):
        """Test svd_circuits with specified batch_size."""
        # Create test data
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)

        # Mock SVD result
        mock_u = th.rand(5, 5)  # Smaller due to batch_size=5
        mock_s = th.tensor([0.9, 0.8, 0.5, 0.3, 0.1])
        mock_vh = th.rand(5, 12)  # 12 = 3*4 (layers * experts)

        # Set up patches
        monkeypatch.setattr("exp.svd_circuits.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr("exp.svd_circuits.FIGURE_DIR", str(temp_dir))

        with (
            patch(
                "exp.svd_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ),
            patch(
                "torch.linalg.svd",
                return_value=(mock_u, mock_s, mock_vh),
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ),
            patch(
                "matplotlib.pyplot.close",
            ),
        ):
            # Run the function with batch_size=5
            svd_circuits(batch_size=5, num_circuits=3, device="cpu")

            # Check that output file was created
            output_file = os.path.join(temp_dir, "svd_circuits.pt")
            assert os.path.exists(output_file)

            # Load and verify the output
            saved_data = th.load(output_file)
            assert th.equal(saved_data["circuits"], mock_vh[:3, :])

    def test_svd_circuits_with_device(self, temp_dir, monkeypatch):
        """Test svd_circuits with specified device."""
        # Create test data
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.bool)

        # Mock SVD result
        mock_u = th.rand(10, 10)
        mock_s = th.tensor([0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
        mock_vh = th.rand(10, 12)

        # Set up patches
        monkeypatch.setattr("exp.svd_circuits.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr("exp.svd_circuits.FIGURE_DIR", str(temp_dir))

        with (
            patch(
                "exp.svd_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ) as mock_load,
            patch(
                "torch.linalg.svd",
                return_value=(mock_u, mock_s, mock_vh),
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ),
            patch(
                "matplotlib.pyplot.close",
            ),
        ):
            # Run the function with device="cuda"
            svd_circuits(device="cuda")

            # Check that load_activations_and_topk was called with device="cuda"
            mock_load.assert_called_once_with(device="cuda")

    def test_svd_circuits_with_real_svd(self, temp_dir, monkeypatch):
        """Test svd_circuits with real SVD computation."""
        # Create test data with known patterns
        mock_activated_experts = th.zeros(10, 3, 4, dtype=th.float32)
        # Create two distinct patterns
        mock_activated_experts[0:5, 0, 0] = (
            1.0  # First 5 samples activate expert 0 in layer 0
        )
        mock_activated_experts[5:10, 1, 2] = (
            1.0  # Last 5 samples activate expert 2 in layer 1
        )

        # Set up patches
        monkeypatch.setattr("exp.svd_circuits.OUTPUT_DIR", str(temp_dir))
        monkeypatch.setattr("exp.svd_circuits.FIGURE_DIR", str(temp_dir))

        with (
            patch(
                "exp.svd_circuits.load_activations_and_topk",
                return_value=(mock_activated_experts, 2),
            ),
            patch(
                "matplotlib.pyplot.savefig",
            ),
            patch(
                "matplotlib.pyplot.close",
            ),
        ):
            # Run the function
            svd_circuits(num_circuits=2, device="cpu")

            # Check that output file was created
            output_file = os.path.join(temp_dir, "svd_circuits.pt")
            assert os.path.exists(output_file)

            # Load and verify the output
            saved_data = th.load(output_file)
            circuits = saved_data["circuits"]

            # Check that we have 2 circuits
            assert circuits.shape[0] == 2

            # The first two singular vectors should capture the two patterns we created
            # Check that the first circuit has high values for expert 0 in layer 0
            # and the second circuit has high values for expert 2 in layer 1
            # (or vice versa, since SVD ordering might vary)
            circuit1 = circuits[0].view(3, 4)
            circuit2 = circuits[1].view(3, 4)

            # At least one of the circuits should have a high value for expert 0 in layer 0
            assert (abs(circuit1[0, 0]) > 0.1) or (abs(circuit2[0, 0]) > 0.1)

            # At least one of the circuits should have a high value for expert 2 in layer 1
            assert (abs(circuit1[1, 2]) > 0.1) or (abs(circuit2[1, 2]) > 0.1)
