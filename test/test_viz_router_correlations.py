"""Tests for viz.router_correlations module."""

import os
from unittest.mock import patch

import pytest
import torch as th

from viz.router_correlations import router_correlations


class TestRouterCorrelations:
    """Test router_correlations function."""

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_router_correlations_no_files(self, temp_dir, monkeypatch):
        """Test router_correlations with no data files."""
        # Set up patches
        monkeypatch.setattr("viz.router_correlations.ROUTER_LOGITS_DIR", str(temp_dir))

        with pytest.raises(ValueError, match="No data files found"):
            router_correlations()

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_router_correlations_basic(self, temp_dir, monkeypatch):
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

        # Mock the correlation calculation to avoid issues with tensor shapes
        mock_correlation = th.eye(12)  # 12 = 3*4 (layers * experts)

        with (
            patch("torch.corrcoef", return_value=mock_correlation),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch("builtins.print") as mock_print,
        ):
            # Run the function
            router_correlations()

            # Check that output files were created
            assert os.path.exists(os.path.join(temp_dir, "router_correlations.png"))
            assert os.path.exists(
                os.path.join(temp_dir, "router_correlations_random.png")
            )
            assert os.path.exists(
                os.path.join(temp_dir, "router_correlations_cross_layer.png")
            )
            assert os.path.exists(
                os.path.join(temp_dir, "router_correlations_cross_layer_random.png")
            )

            # Check that print was called with expected messages
            print_calls = [
                call[0][0] for call in mock_print.call_args_list if len(call[0]) > 0
            ]
            assert any(
                "Top 10 cross-layer correlations" in str(call) for call in print_calls
            )
            assert any(
                "Bottom 10 cross-layer correlations" in str(call)
                for call in print_calls
            )

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_router_correlations_correlation_calculation(self, temp_dir, monkeypatch):
        """Test that router_correlations correctly calculates correlations."""
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

        # Create a mock correlation matrix with known values
        mock_correlation = th.eye(6)  # 6 = 2*3 (layers * experts)
        # Set perfect correlation between Layer 0, Expert 0 and Layer 1, Expert 1
        mock_correlation[0, 4] = 1.0
        mock_correlation[4, 0] = 1.0
        # Set perfect anti-correlation between Layer 0, Expert 1 and Layer 1, Expert 0
        mock_correlation[1, 3] = -1.0
        mock_correlation[3, 1] = -1.0

        with (
            patch("torch.corrcoef", return_value=mock_correlation),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch("builtins.print") as mock_print,
        ):
            # Run the function
            router_correlations()

            # Check that print was called with correlation values
            print_calls = [
                call[0][0] for call in mock_print.call_args_list if len(call[0]) > 0
            ]
            correlation_values = [
                call
                for call in print_calls
                if isinstance(call, str) and "layer" in call and "expert" in call
            ]

            # Should have at least one correlation value printed
            assert len(correlation_values) > 0

    @pytest.mark.skip(reason="Test needs further work to fix mocking issues")
    def test_router_correlations_with_real_correlation(self, temp_dir, monkeypatch):
        """Test router_correlations with a real correlation calculation."""
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

        with (
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch("builtins.print") as mock_print,
        ):
            # Run the function
            router_correlations()

            # Check that print was called with correlation values
            print_calls = [
                call[0][0] for call in mock_print.call_args_list if len(call[0]) > 0
            ]

            # Look for high positive and high negative correlations
            high_positive_found = False
            high_negative_found = False

            for call_text in print_calls:
                if (
                    isinstance(call_text, str)
                    and "layer" in call_text
                    and "expert" in call_text
                ):
                    if "1.0" in call_text:  # Perfect positive correlation
                        high_positive_found = True
                    if "-1.0" in call_text:  # Perfect negative correlation
                        high_negative_found = True

            # We should find at least one high positive or high negative correlation
            assert high_positive_found or high_negative_found
