"""Test utilities and helper functions."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import torch as th


def create_mock_file_structure(base_dir: Path, files: dict[str, Any]) -> None:
    """Create a mock file structure for testing.

    Args:
        base_dir: Base directory to create files in
        files: Dictionary mapping file paths to content (torch tensors will be saved with torch.save)
    """
    for file_path, content in files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, th.Tensor | dict):
            th.save(content, full_path)
        else:
            full_path.write_text(str(content))


def assert_tensor_shape_and_type(
    tensor: th.Tensor, expected_shape: tuple, expected_dtype: th.dtype = None
) -> None:
    """Assert tensor has expected shape and optionally dtype."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        )


def assert_tensor_close(
    actual: th.Tensor, expected: th.Tensor, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    """Assert two tensors are close within tolerance."""
    assert th.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Tensors not close:\nActual: {actual}\nExpected: {expected}"
    )


def create_sample_router_data(
    batch_size: int = 4,
    num_layers: int = 3,
    num_experts: int = 8,
    topk: int = 2,
    device: str = "cpu",
) -> dict[str, th.Tensor]:
    """Create sample router activation data for testing."""
    router_logits = th.randn(batch_size, num_layers, num_experts, device=device)

    # Create topk activations
    topk_indices = th.topk(router_logits, k=topk, dim=2).indices
    activated_experts = th.zeros_like(router_logits, dtype=th.bool, device=device)
    activated_experts.scatter_(2, topk_indices, True)

    return {
        "router_logits": router_logits,
        "activated_experts": activated_experts,
        "topk_indices": topk_indices,
        "topk": topk,
    }


def create_sample_circuits(
    num_circuits: int = 3,
    num_layers: int = 3,
    num_experts: int = 8,
    sparsity: float = 0.2,
    device: str = "cpu",
) -> th.Tensor:
    """Create sample circuit tensor for testing."""
    circuits = th.rand(num_circuits, num_layers, num_experts, device=device) < sparsity
    return circuits


def create_sample_circuit_logits(
    num_circuits: int = 3,
    num_layers: int = 3,
    num_experts: int = 8,
    device: str = "cpu",
) -> th.Tensor:
    """Create sample circuit logits for testing."""
    return th.randn(num_circuits, num_layers, num_experts, device=device)


class MockFile:
    """Mock file object for testing file operations."""

    def __init__(self, content: Any = None):
        self.content = content
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self):
        return self.content

    def write(self, content):
        self.content = content


class TensorGenerator:
    """Utility class for generating test tensors with specific properties."""

    @staticmethod
    def boolean_tensor(
        shape: tuple, true_probability: float = 0.3, device: str = "cpu"
    ) -> th.Tensor:
        """Generate a boolean tensor with specified probability of True values."""
        return th.rand(shape, device=device) < true_probability

    @staticmethod
    def normalized_tensor(
        shape: tuple, dim: int = -1, device: str = "cpu"
    ) -> th.Tensor:
        """Generate a tensor normalized along specified dimension."""
        tensor = th.randn(shape, device=device)
        return th.nn.functional.softmax(tensor, dim=dim)

    @staticmethod
    def correlation_matrix(size: int, device: str = "cpu") -> th.Tensor:
        """Generate a valid correlation matrix."""
        # Generate random matrix and make it symmetric positive definite
        A = th.randn(size, size, device=device)
        corr = A @ A.T
        # Normalize to correlation matrix
        std = th.sqrt(th.diag(corr))
        corr = corr / (std.unsqueeze(0) * std.unsqueeze(1))
        return corr

    @staticmethod
    def sparse_tensor(
        shape: tuple, sparsity: float = 0.1, device: str = "cpu"
    ) -> th.Tensor:
        """Generate a sparse tensor with specified sparsity level."""
        tensor = th.randn(shape, device=device)
        mask = th.rand(shape, device=device) < sparsity
        return tensor * mask


def validate_device_map(
    device_map: dict[str, str], expected_devices: list[str]
) -> None:
    """Validate that a device map contains expected devices and proper structure."""
    assert isinstance(device_map, dict), "Device map must be a dictionary"

    # Check that all values are valid devices
    for key, device in device_map.items():
        assert isinstance(key, str), f"Device map key must be string, got {type(key)}"
        assert (
            device in expected_devices or device == "cpu" or isinstance(device, int)
        ), f"Invalid device {device} for key {key}"

    # Check for expected key patterns
    expected_patterns = ["model.embed_tokens.weight", "model.layers.", "lm_head.weight"]

    found_patterns = set()
    for key in device_map:
        for pattern in expected_patterns:
            if pattern in key:
                found_patterns.add(pattern)
                break

    assert len(found_patterns) > 0, (
        "Device map should contain expected model component keys"
    )


def create_temporary_data_files(temp_dir: Path, num_files: int = 3) -> list[Path]:
    """Create temporary data files for testing file loading operations."""
    file_paths = []

    for i in range(num_files):
        file_path = temp_dir / f"{i}.pt"
        data = {
            "topk": 2,
            "router_logits": th.randn(5, 3, 8),  # 5 tokens, 3 layers, 8 experts
        }
        th.save(data, file_path)
        file_paths.append(file_path)

    return file_paths


def mock_huggingface_repo_refs(branches: list[str]) -> MagicMock:
    """Create a mock for huggingface_hub.list_repo_refs."""
    mock_refs = MagicMock()
    mock_branches = []

    for branch_name in branches:
        mock_branch = MagicMock()
        mock_branch.name = branch_name
        mock_branches.append(mock_branch)

    mock_refs.branches = mock_branches
    return mock_refs


def assert_valid_iou_score(iou: th.Tensor) -> None:
    """Assert that IoU scores are valid (between 0 and 1)."""
    assert th.all(iou >= 0.0), f"IoU scores must be >= 0, got min: {th.min(iou)}"
    assert th.all(iou <= 1.0), f"IoU scores must be <= 1, got max: {th.max(iou)}"


def assert_valid_loss(loss: th.Tensor) -> None:
    """Assert that loss values are valid (non-negative, finite)."""
    assert th.all(th.isfinite(loss)), "Loss values must be finite"
    assert th.all(loss >= 0.0), (
        f"Loss values must be non-negative, got min: {th.min(loss)}"
    )


def create_mock_pca_result(n_components: int = 2, n_samples: int = 100) -> th.Tensor:
    """Create mock PCA transformation result."""
    return th.randn(n_samples, n_components)
