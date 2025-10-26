"""Tests for router correlations visualization."""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch


@pytest.mark.skip(reason="Needs more robust mocking")
def test_router_correlations_empty_directory():
    """Test router_correlations with an empty directory."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        (
            patch(
                "exp.get_experiment_dir",
                return_value=os.path.join(temp_dir, "test_experiment"),
            ),
            patch("exp.get_router_logits_dir", return_value=str(temp_dir)),
        ),
    ):
        from viz.router_correlations import router_correlations

        with pytest.raises(ValueError, match="No data files found"):
            router_correlations(experiment_name="test_experiment")


@pytest.mark.skip(reason="Needs more robust mocking")
def test_router_correlations_basic():
    """Test basic functionality of router_correlations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        batch_size = 10
        num_layers = 2
        num_experts = 8
        topk = 2

        # Create router logits files
        router_logits_dir = os.path.join(temp_dir, "router_logits")
        os.makedirs(router_logits_dir, exist_ok=True)

        # Create a single router logits file
        router_logits = torch.randn(batch_size, num_layers, num_experts)
        torch.save(
            {"router_logits": router_logits, "topk": topk},
            os.path.join(router_logits_dir, "0.pt"),
        )

        # Create experiment and figure directories
        experiment_dir = os.path.join(temp_dir, "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        figure_dir = os.path.join(temp_dir, "fig", "test_experiment")
        os.makedirs(figure_dir, exist_ok=True)

        # Patch the directory functions
        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.get_router_logits_dir", return_value=router_logits_dir),
            patch("viz.get_figure_dir", return_value=figure_dir),
        ):
            # Import here to avoid module-level binding issues
            from viz.router_correlations import router_correlations

            # Run the function
            router_correlations(experiment_name="test_experiment")

            # Check that the output files were created
            assert os.path.exists(os.path.join(figure_dir, "router_correlations.png"))
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_random.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer_random.png")
            )


@pytest.mark.skip(reason="Needs more robust mocking")
def test_router_correlations_multiple_files():
    """Test router_correlations with multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        batch_size = 5
        num_layers = 2
        num_experts = 8
        topk = 2

        # Create router logits files
        router_logits_dir = os.path.join(temp_dir, "router_logits")
        os.makedirs(router_logits_dir, exist_ok=True)

        # Create multiple router logits files
        for i in range(3):
            router_logits = torch.randn(batch_size, num_layers, num_experts)
            torch.save(
                {"router_logits": router_logits, "topk": topk},
                os.path.join(router_logits_dir, f"{i}.pt"),
            )

        # Create experiment and figure directories
        experiment_dir = os.path.join(temp_dir, "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        figure_dir = os.path.join(temp_dir, "fig", "test_experiment")
        os.makedirs(figure_dir, exist_ok=True)

        # Patch the directory functions
        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.get_router_logits_dir", return_value=router_logits_dir),
            patch("viz.get_figure_dir", return_value=figure_dir),
        ):
            # Import here to avoid module-level binding issues
            from viz.router_correlations import router_correlations

            # Run the function
            router_correlations(experiment_name="test_experiment")

            # Check that the output files were created
            assert os.path.exists(os.path.join(figure_dir, "router_correlations.png"))
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_random.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer_random.png")
            )


@pytest.mark.skip(reason="Needs more robust mocking")
def test_router_correlations_with_tokens():
    """Test router_correlations with tokens."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        batch_size = 5
        num_layers = 2
        num_experts = 8
        topk = 2

        # Create router logits files
        router_logits_dir = os.path.join(temp_dir, "router_logits")
        os.makedirs(router_logits_dir, exist_ok=True)

        # Create a single router logits file with tokens
        router_logits = torch.randn(batch_size, num_layers, num_experts)
        tokens = ["token1", "token2", "token3", "token4", "token5"]
        torch.save(
            {"router_logits": router_logits, "topk": topk, "tokens": tokens},
            os.path.join(router_logits_dir, "0.pt"),
        )

        # Create experiment and figure directories
        experiment_dir = os.path.join(temp_dir, "test_experiment")
        os.makedirs(experiment_dir, exist_ok=True)
        figure_dir = os.path.join(temp_dir, "fig", "test_experiment")
        os.makedirs(figure_dir, exist_ok=True)

        # Patch the directory functions
        with (
            patch("exp.get_experiment_dir", return_value=experiment_dir),
            patch("exp.get_router_logits_dir", return_value=router_logits_dir),
            patch("viz.get_figure_dir", return_value=figure_dir),
        ):
            # Import here to avoid module-level binding issues
            from viz.router_correlations import router_correlations

            # Run the function
            router_correlations(experiment_name="test_experiment")

            # Check that the output files were created
            assert os.path.exists(os.path.join(figure_dir, "router_correlations.png"))
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_random.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer.png")
            )
            assert os.path.exists(
                os.path.join(figure_dir, "router_correlations_cross_layer_random.png")
            )
