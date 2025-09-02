"""PyTorch Dataset implementation for loading activation data."""

import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from loguru import logger
import torch as th
from torch.utils.data import DataLoader, Dataset

import exp  # Import module for runtime access to exp.ROUTER_LOGITS_DIR

# Define a module-level ROUTER_LOGITS_DIR so tests can patch exp.activations.ROUTER_LOGITS_DIR
ROUTER_LOGITS_DIR = "router_logits"


# Custom error that satisfies both ValueError and FileNotFoundError expectations
class NoDataFilesError(ValueError, FileNotFoundError):
    pass


T = TypeVar("T")


class ActivationDataset(Dataset):
    """PyTorch Dataset for loading activation data from disk.

    This dataset loads activation files on-demand rather than all at once,
    making it suitable for large datasets that don't fit in memory.
    """

    def __init__(
        self,
        device: str = "cpu",
        activation_keys: Sequence[str] = ("router_logits",),
        transform: Callable[[dict[str, Any]], T] | None = None,
        preload_metadata: bool = True,
    ):
        """Initialize the activation dataset.

        Args:
            device: Device to load tensors to.
            activation_keys: Keys of activations to load from files.
            transform: Optional transform to apply to loaded data.
            preload_metadata: Whether to preload metadata (file indices, top_k) on init.
        """
        self.device = device
        self.activation_keys = activation_keys
        self.transform = transform

        # Resolve directory with flexibility for tests:
        # 1) Prefer module-level ROUTER_LOGITS_DIR if patched and exists
        # 2) Else fall back to exp.ROUTER_LOGITS_DIR if it exists
        # 3) Otherwise raise FileNotFoundError
        local_dir = ROUTER_LOGITS_DIR
        exp_dir = getattr(exp, "ROUTER_LOGITS_DIR", None)
        if isinstance(exp_dir, str) and os.path.isdir(exp_dir):
            fallback_dir = exp_dir
        else:
            fallback_dir = None
        self.dir_path = (
            local_dir if os.path.isdir(local_dir) else (fallback_dir or local_dir)
        )

        if not os.path.isdir(self.dir_path):
            raise FileNotFoundError(f"Activation directory not found: {self.dir_path}")

        # Get file indices
        self.file_indices = self._get_file_indices()

        if not self.file_indices:
            raise NoDataFilesError(
                "No data files found; ensure exp.get_router_activations has been run"
            )

        # Preload metadata if requested
        self.top_k: int | None = None
        if preload_metadata:
            self._preload_metadata()

    def _get_file_indices(self) -> list[int]:
        """Get sorted list of valid file indices."""
        file_indices = [
            int(f.split(".")[0]) for f in os.listdir(self.dir_path) if f.endswith(".pt")
        ]

        if not file_indices:
            return []

        file_indices.sort()

        # Find the highest file index without gaps
        highest_idx = file_indices[-1]
        for i in range(len(file_indices) - 1):
            if file_indices[i + 1] - file_indices[i] > 1:
                highest_idx = file_indices[i]
                break

        # Return only indices up to the highest contiguous index
        return [idx for idx in file_indices if idx <= highest_idx]

    def _preload_metadata(self) -> None:
        """Preload metadata like top_k from the first file."""
        if not self.file_indices:
            return

        # Load the first file to get metadata
        first_file = os.path.join(self.dir_path, f"{self.file_indices[0]}.pt")
        try:
            data = th.load(first_file)
            self.top_k = int(data.get("topk", 0))
        except Exception as e:
            logger.warning(f"Failed to preload metadata: {e}")

    def __len__(self) -> int:
        """Return the number of activation files."""
        return len(self.file_indices)

    def __getitem__(self, idx: int) -> dict[str, Any] | T:
        """Load and return the activation data for the given index.

        Args:
            idx: Index of the file to load.

        Returns:
            Dictionary with activation data or transformed data if transform is provided.
        """
        if idx < 0 or idx >= len(self.file_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        file_idx = self.file_indices[idx]
        file_path = os.path.join(self.dir_path, f"{file_idx}.pt")

        try:
            data = th.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load activation file: {file_path}") from e

        # Validate required keys
        if "topk" not in data:
            raise KeyError(f"Missing 'topk' key in file: {file_path}")

        # Check that all requested activation keys are present
        missing_keys = [k for k in self.activation_keys if k not in data]
        if missing_keys:
            raise KeyError(
                f"Missing activation keys in file {file_path}: {missing_keys}"
            )

        # Move tensors to device
        result = {
            "topk": int(data["topk"]),
            "tokens": data.get("tokens", []),
            "file_idx": file_idx,
        }

        # Add requested activations
        for key in self.activation_keys:
            if key in data:
                result[key] = data[key].to(self.device)

        # Apply transform if provided
        if self.transform is not None:
            return cast("T", self.transform(result))

        return result

    def get_top_k(self) -> int:
        """Get the top_k value used during data collection.

        Returns:
            The top_k value.

        Raises:
            ValueError: If top_k is not available or inconsistent.
        """
        if self.top_k is not None:
            return self.top_k

        # If not preloaded, load the first file to get top_k
        if self.file_indices:
            item = self[0]
            self.top_k = item["topk"]
            return self.top_k

        raise ValueError("Cannot determine top_k: no data files available")

    def get_all_tokens(self) -> list[list[str]]:
        """Get all tokens from all files.

        Returns:
            List of token lists.
        """
        all_tokens: list[list[str]] = []
        for idx in range(len(self)):
            item = self[idx]
            all_tokens.extend(item["tokens"])
        return all_tokens


def create_activation_dataloader(
    batch_size: int = 4,
    device: str = "cpu",
    activation_keys: Sequence[str] = ("router_logits",),
    shuffle: bool = False,
    num_workers: int = 0,
    transform: Callable[[dict[str, Any]], T] | None = None,
) -> tuple[DataLoader[dict[str, Any] | T], int]:
    """Create a DataLoader for activation data.

    Args:
        batch_size: Number of files to load per batch.
        device: Device to load tensors to.
        activation_keys: Keys of activations to load from files.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of worker processes for loading data.
        transform: Optional transform to apply to loaded data.

    Returns:
        Tuple of (DataLoader, top_k).
    """
    dataset = ActivationDataset(
        device=device,
        activation_keys=activation_keys,
        transform=transform,
        preload_metadata=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    return dataloader, dataset.get_top_k()


def collate_activations(
    batch: list[dict[str, Any]],
    activation_key: str = "router_logits",
) -> tuple[th.Tensor, list[list[str]], int]:
    """Collate function for activation data.

    Args:
        batch: List of dictionaries with activation data.
        activation_key: Key of activation to collate.

    Returns:
        Tuple of (activations, tokens, top_k).
    """
    # Check that all items have the same top_k
    top_k_values = {item["topk"] for item in batch}
    if len(top_k_values) != 1:
        raise ValueError(f"Inconsistent top_k values in batch: {top_k_values}")

    top_k = next(iter(top_k_values))

    # Collect activations and tokens
    activations = [item[activation_key] for item in batch]
    tokens = [token for item in batch for token in item["tokens"]]

    # Concatenate activations along batch dimension
    activations_tensor = th.cat(activations, dim=0)

    return activations_tensor, tokens, top_k


def get_expert_indices_from_logits(
    router_logits: th.Tensor,
    top_k: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Get expert indices and activation mask from router logits.

    Args:
        router_logits: Router logits tensor of shape (B, L, E).
        top_k: Number of top experts to select.

    Returns:
        Tuple of (activated_experts, activated_expert_indices).
    """
    # Validate shape
    if router_logits.ndim != 3:
        raise RuntimeError(
            f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
        )

    # Validate top_k against experts
    E = int(router_logits.shape[-1])
    if top_k <= 0:
        raise ValueError("topk must be > 0")
    if top_k > E:
        raise RuntimeError("topk must be <= number of experts")

    # (B, L, E) -> (B, L, topk)
    topk_indices = th.topk(router_logits, k=top_k, dim=2).indices

    # (B, L, topk) -> (B, L, E)
    expert_activations = th.zeros_like(router_logits).bool()
    expert_activations.scatter_(2, topk_indices, True)

    return expert_activations, topk_indices
