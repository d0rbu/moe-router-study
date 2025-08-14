"""Tests for the expert importances visualization."""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch as th

from viz.expert_importances import (
    READER_COMPONENTS,
    WRITER_COMPONENTS,
    expert_importances,
)


@pytest.fixture
def mock_expert_importance_data():
    """Create mock expert importance data for testing."""
    entries = []

    # Create test data for 2 layers, 2 experts
    for layer_idx in range(2):
        for expert_idx in range(2):
            # Add reader components
            entries.extend([
                {
                    "layer_idx": layer_idx,
                    "component": component,
                    "expert_idx": expert_idx,
                    "role": "reader",
                    "l2": 0.5 + layer_idx * 0.1 + expert_idx * 0.2,
                    "model_name": "test_model",
                    "checkpoint_idx": 0,
                } for component in READER_COMPONENTS
            ])

            # Add writer components
            entries.extend([
                {
                    "layer_idx": layer_idx,
                    "component": component,
                    "expert_idx": expert_idx,
                    "role": "writer",
                    "l2": 0.3 + layer_idx * 0.1 + expert_idx * 0.2,
                    "model_name": "test_model",
                    "checkpoint_idx": 0,
                } for component in WRITER_COMPONENTS
            ])

    return entries


@pytest.fixture
def temp_data_file(mock_expert_importance_data):
    """Create a temporary file with mock expert importance data."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        th.save(mock_expert_importance_data, tmp.name)
        yield tmp.name

    # Clean up the temporary file after the test
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


@patch("matplotlib.pyplot.show")
def test_expert_importances_loads_data(mock_show, temp_data_file):
    """Test that expert_importances can load and process data."""
    # Run the visualization function with the test data
    expert_importances(
        data_path=temp_data_file,
        model_name="test_model",
        checkpoint_idx=0,
        initial_layer_idx=0,
        initial_expert_idx=0,
    )

    # Verify that plt.show() was called, indicating the visualization was created
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_expert_importances_with_filters(mock_show, temp_data_file):
    """Test that expert_importances correctly applies filters."""
    # Run with specific model_name and checkpoint_idx filters
    expert_importances(
        data_path=temp_data_file,
        model_name="test_model",
        checkpoint_idx=0,
        initial_layer_idx=1,
        initial_expert_idx=1,
    )

    # Verify that plt.show() was called
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_expert_importances_with_custom_percentile(mock_show, temp_data_file):
    """Test that expert_importances works with custom normalization percentile."""
    # Run with a custom normalization percentile
    expert_importances(
        data_path=temp_data_file,
        normalize_percentile=90.0,
    )

    # Verify that plt.show() was called
    mock_show.assert_called_once()


@pytest.mark.parametrize(
    "invalid_path",
    [
        "nonexistent_file.pt",
        "/path/does/not/exist/data.pt",
    ]
)
def test_expert_importances_file_not_found(invalid_path):
    """Test that expert_importances raises FileNotFoundError for invalid paths."""
    with pytest.raises(FileNotFoundError):
        expert_importances(data_path=invalid_path)


@patch("torch.load")
def test_expert_importances_empty_data(mock_load, temp_data_file):
    """Test that expert_importances raises ValueError for empty data."""
    # Mock torch.load to return empty list
    mock_load.return_value = []

    with pytest.raises(ValueError, match="No entries found"):
        expert_importances(
            data_path=temp_data_file,
            model_name="nonexistent_model",
        )
