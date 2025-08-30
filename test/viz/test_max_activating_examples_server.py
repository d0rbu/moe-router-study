import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch as th

from viz.max_activating_examples_server import (
    compute_max_activating_examples,
    generate_random_mask,
    load_circuits,
    save_circuits,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_experiment_dir(temp_dir):
    """Create a mock experiment directory."""
    experiment_dir = os.path.join(temp_dir, "test_experiment")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def test_save_and_load_circuits(mock_experiment_dir):
    """Test saving and loading circuits."""
    # Create test data
    circuits = th.ones((2, 3, 4))
    names = ["test1", "test2"]
    circuits_dict = {"circuits": circuits, "names": names}
    
    circuits_path = os.path.join(mock_experiment_dir, "saved_circuits.pt")

    # Mock get_experiment_dir to return our test directory
    with patch("exp.get_experiment_dir", return_value=mock_experiment_dir):
        # Save circuits
        save_circuits(circuits_dict, experiment_name="test_experiment")

        # Load circuits
        loaded_dict = load_circuits(experiment_name="test_experiment")

    # Check that the loaded data matches the original
    assert th.allclose(loaded_dict["circuits"], circuits)
    assert loaded_dict["names"] == names


def test_load_circuits_nonexistent(mock_experiment_dir):
    """Test loading circuits when the file doesn't exist."""
    # Mock get_experiment_dir to return our test directory
    with patch("exp.get_experiment_dir", return_value=mock_experiment_dir):
        # Load non-existent circuits
        loaded_dict = load_circuits(experiment_name="test_experiment")

    # Check that we get an empty dictionary with the expected structure
    assert loaded_dict["circuits"].shape == (0, 0, 0)
    assert loaded_dict["names"] == []


def test_generate_random_mask():
    """Test generating a random mask."""
    num_layers = 3
    num_experts = 5
    top_k = 2

    # Generate a random mask
    mask = generate_random_mask(num_layers, num_experts, top_k)

    # Check shape
    assert mask.shape == (num_layers, num_experts)

    # Check that each layer has exactly top_k ones
    for layer in range(num_layers):
        assert th.sum(mask[layer]) == top_k

    # Check that all values are either 0 or 1
    assert set(mask.flatten().tolist()) <= {0.0, 1.0}


def test_compute_max_activating_examples():
    """Test computing max activating examples."""
    # Create test data
    circuit = th.zeros((2, 3))
    circuit[0, 0] = 1.0
    circuit[1, 1] = 1.0

    # Create a token_topk_mask where some tokens activate the circuit
    token_topk_mask = th.zeros((5, 2, 3), dtype=th.bool)
    token_topk_mask[0, 0, 0] = True  # Token 0 activates circuit[0, 0]
    token_topk_mask[1, 1, 1] = True  # Token 1 activates circuit[1, 1]
    token_topk_mask[2, 0, 0] = True  # Token 2 activates circuit[0, 0]
    token_topk_mask[2, 1, 1] = True  # Token 2 also activates circuit[1, 1]
    token_topk_mask[3, 0, 1] = True  # Token 3 doesn't activate the circuit

    top_k = 1
    top_n = 3

    # Compute max activating examples
    norm_scores, top_indices = compute_max_activating_examples(
        circuit, token_topk_mask, top_k, device="cpu", top_n=top_n
    )

    # Check shape of outputs
    assert norm_scores.shape == (5,)
    assert len(top_indices) == top_n

    # Check that the scores are as expected
    # Token 2 should have the highest score (activates both circuit elements)
    # Tokens 0 and 1 should have the next highest scores (each activates one element)
    # Token 3 should have a score of 0 (doesn't activate any circuit element)
    expected_order = [2, 0, 1]
    assert top_indices[:3] == expected_order

    # Check normalization
    # Each token can activate at most top_k * num_layers = 1 * 2 = 2 elements
    # Token 2 activates 2 elements, so its score should be 1.0
    # Tokens 0 and 1 each activate 1 element, so their scores should be 0.5
    assert norm_scores[2] == 1.0
    assert norm_scores[0] == 0.5
    assert norm_scores[1] == 0.5
    assert norm_scores[3] == 0.0


@pytest.mark.parametrize("input_type", ["list", "numpy", "tensor"])
def test_compute_max_activating_examples_input_types(input_type):
    """Test compute_max_activating_examples with different input types."""
    # Create test data
    if input_type == "list":
        circuit = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    elif input_type == "numpy":
        circuit = np.zeros((2, 3))
        circuit[0, 0] = 1.0
        circuit[1, 1] = 1.0
    else:  # tensor
        circuit = th.zeros((2, 3))
        circuit[0, 0] = 1.0
        circuit[1, 1] = 1.0

    # Create a token_topk_mask
    token_topk_mask = th.zeros((5, 2, 3), dtype=th.bool)
    token_topk_mask[0, 0, 0] = True

    # Compute max activating examples
    norm_scores, top_indices = compute_max_activating_examples(
        circuit, token_topk_mask, top_k=1, device="cpu", top_n=3
    )

    # Check that we get valid outputs regardless of input type
    assert isinstance(norm_scores, th.Tensor)
    assert isinstance(top_indices, list)
    assert norm_scores.shape == (5,)
    assert len(top_indices) <= 3

