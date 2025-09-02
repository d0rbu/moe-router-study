from collections.abc import Iterator
import os
from typing import Any

from loguru import logger
import torch as th
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import exp  # Import module for runtime access to exp.ROUTER_LOGITS_DIR

# Define a module-level ROUTER_LOGITS_DIR so tests can patch exp.activations.ROUTER_LOGITS_DIR
ROUTER_LOGITS_DIR = "router_logits"


# Custom error that satisfies both ValueError and FileNotFoundError expectations
class NoDataFilesError(ValueError, FileNotFoundError):
    pass


class ActivationsDataset(Dataset):
    """PyTorch Dataset for loading activation files on demand.

    This dataset loads each activation file only when requested, which allows
    handling large datasets that don't fit in memory.
    """

    def __init__(self, dir_path: str, device: str = "cpu"):
        """Initialize the dataset.

        Args:
            dir_path: Path to the directory containing activation files
            device: Device to load tensors to
        """
        self.dir_path = dir_path
        self.device = device

        # Check if directory exists
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Activation directory not found: {dir_path}")

        # Get all file indices
        self.file_indices = [
            int(f.split(".")[0]) for f in os.listdir(dir_path) if f.endswith(".pt")
        ]

        # Check if there are any files
        if not self.file_indices:
            raise NoDataFilesError("No data files found in directory")

        self.file_indices.sort()

        # Find the highest contiguous file index
        self.highest_file_idx = self.file_indices[-1]
        for i in range(len(self.file_indices) - 1):
            if self.file_indices[i + 1] - self.file_indices[i] > 1:
                self.highest_file_idx = self.file_indices[i]
                break

        # Create a mapping from dataset index to file index
        self.valid_indices = self.file_indices[
            : self.file_indices.index(self.highest_file_idx) + 1
        ]

        # Cache for top_k value
        self._top_k = None

    def __len__(self) -> int:
        """Return the number of files in the dataset."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a specific activation file.

        Args:
            idx: Index of the file to load

        Returns:
            Dictionary containing the loaded activation data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )

        file_idx = self.valid_indices[idx]
        file_path = os.path.join(self.dir_path, f"{file_idx}.pt")

        try:
            output = th.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load router logits file: {file_path}") from e

        # Required keys
        if "topk" not in output or "router_logits" not in output:
            missing = [k for k in ("topk", "router_logits") if k not in output]
            raise KeyError(f"Missing keys in logits file: {missing}")

        # Normalize to python int
        file_topk = int(output["topk"])  # type: ignore[call-overload]

        # Store top_k if not already set
        if self._top_k is None:
            self._top_k = file_topk
        elif file_topk != self._top_k:
            raise KeyError(
                f"Inconsistent topk across files: saw {file_topk} then {self._top_k}"
            )

        # Move tensors to device
        for key, value in output.items():
            if isinstance(value, th.Tensor):
                output[key] = value.to(self.device)

        # Process router logits to get indices and activations
        router_logits: th.Tensor = output["router_logits"]

        # Validate shape
        if router_logits.ndim != 3:
            raise RuntimeError(
                f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
            )

        # Validate top_k against experts
        E = int(router_logits.shape[-1])
        if self._top_k <= 0:
            raise ValueError("topk must be > 0")
        if self._top_k > E:
            raise RuntimeError("topk must be <= number of experts")

        # (B, L, E) -> (B, L, topk)
        topk_indices = th.topk(router_logits, k=self._top_k, dim=2).indices

        # (B, L, topk) -> (B, L, E)
        expert_activations = th.zeros_like(router_logits, device=self.device).bool()
        expert_activations.scatter_(2, topk_indices, True)

        # Add processed data to output
        output["activated_experts"] = expert_activations
        output["activated_expert_indices"] = topk_indices

        return output

    @property
    def top_k(self) -> int:
        """Get the top_k value from the dataset.

        This will load the first file to determine top_k if it hasn't been loaded yet.
        """
        if self._top_k is None:
            # Load the first file to get top_k
            _ = self[0]
        return self._top_k  # type: ignore[return-value]


def get_activations_dataset(device: str = "cpu") -> ActivationsDataset:
    """Get the activations dataset.

    Args:
        device: Device to load tensors to

    Returns:
        ActivationsDataset instance
    """
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"

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
    dir_path = local_dir if os.path.isdir(local_dir) else (fallback_dir or local_dir)

    return ActivationsDataset(dir_path, device)


def load_activations_indices_tokens_and_topk(
    device: str = "cpu",  # default to CPU to avoid requiring CUDA in tests/CI
) -> tuple[th.Tensor, th.Tensor, list[list[str]], int]:
    """Load boolean activation mask, top-k indices, tokens, and top_k.

    This function loads all data into memory at once, which may not be suitable
    for very large datasets. Consider using the ActivationsDataset directly for
    large datasets.

    Returns:
      - activated_experts: (B, L, E) boolean mask of top-k expert activations
      - activated_expert_indices: (B, L, topk) long indices of selected experts
      - tokens: list[list[str]] tokenized sequences aligned to batch concatenation
      - top_k: int top-k used during collection
    """
    dataset = get_activations_dataset(device)

    activated_experts_collection: list[th.Tensor] = []
    activated_expert_indices_collection: list[th.Tensor] = []
    tokens: list[list[str]] = []

    for i in tqdm(range(len(dataset)), desc="Loading activations", total=len(dataset)):
        data = dataset[i]
        activated_experts_collection.append(data["activated_experts"])
        activated_expert_indices_collection.append(data["activated_expert_indices"])
        file_tokens: list[list[str]] = data.get("tokens", [])
        tokens.extend(file_tokens)

    # (B, L, E)
    activated_experts = th.cat(activated_experts_collection, dim=0)
    # (B, L, topk)
    activated_expert_indices = th.cat(activated_expert_indices_collection, dim=0)

    return activated_experts, activated_expert_indices, tokens, dataset.top_k


def load_activations_and_indices_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, int]:
    activated_experts, activated_expert_indices, _tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, activated_expert_indices, top_k


def load_activations_and_topk(device: str = "cuda") -> tuple[th.Tensor, int]:
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _indices, top_k = load_activations_and_indices_and_topk(
        device=device
    )
    return activated_experts, top_k


def load_activations(device: str = "cuda") -> th.Tensor:
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_tokens_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, list[list[str]], int]:
    activated_experts, _indices, tokens, top_k = (
        load_activations_indices_tokens_and_topk(device=device)
    )
    return activated_experts, tokens, top_k


class ActivationsIterableDataset(IterableDataset):
    """Iterable dataset for streaming activation files.

    This dataset is useful when you need to process files sequentially without
    loading all of them into memory at once.
    """

    def __init__(self, dir_path: str, device: str = "cpu", batch_size: int = 1):
        """Initialize the iterable dataset.

        Args:
            dir_path: Path to the directory containing activation files
            device: Device to load tensors to
            batch_size: Number of files to process at once
        """
        self.dir_path = dir_path
        self.device = device
        self.batch_size = batch_size

        # Check if directory exists
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Activation directory not found: {dir_path}")

        # Get all file indices
        self.file_indices = [
            int(f.split(".")[0]) for f in os.listdir(dir_path) if f.endswith(".pt")
        ]

        # Check if there are any files
        if not self.file_indices:
            raise NoDataFilesError("No data files found in directory")

        self.file_indices.sort()

        # Find the highest contiguous file index
        self.highest_file_idx = self.file_indices[-1]
        for i in range(len(self.file_indices) - 1):
            if self.file_indices[i + 1] - self.file_indices[i] > 1:
                self.highest_file_idx = self.file_indices[i]
                break

        # Create a mapping from dataset index to file index
        self.valid_indices = self.file_indices[
            : self.file_indices.index(self.highest_file_idx) + 1
        ]

        # Cache for top_k value
        self._top_k = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through the activation files.

        Yields:
            Dictionary containing the loaded activation data
        """
        for i in range(0, len(self.valid_indices), self.batch_size):
            batch_indices = self.valid_indices[i : i + self.batch_size]

            batch_data = {
                "activated_experts": [],
                "activated_expert_indices": [],
                "tokens": [],
            }

            for file_idx in batch_indices:
                file_path = os.path.join(self.dir_path, f"{file_idx}.pt")

                try:
                    output = th.load(file_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load router logits file: {file_path}"
                    ) from e

                # Required keys
                if "topk" not in output or "router_logits" not in output:
                    missing = [k for k in ("topk", "router_logits") if k not in output]
                    raise KeyError(f"Missing keys in logits file: {missing}")

                # Normalize to python int
                file_topk = int(output["topk"])  # type: ignore[call-overload]

                # Store top_k if not already set
                if self._top_k is None:
                    self._top_k = file_topk
                elif file_topk != self._top_k:
                    raise KeyError(
                        f"Inconsistent topk across files: saw {file_topk} then {self._top_k}"
                    )

                # Move tensors to device
                router_logits: th.Tensor = output["router_logits"].to(self.device)

                # Validate shape
                if router_logits.ndim != 3:
                    raise RuntimeError(
                        f"Invalid router_logits shape {tuple(router_logits.shape)}; expected (B, L, E)"
                    )

                # Validate top_k against experts
                E = int(router_logits.shape[-1])
                if self._top_k <= 0:
                    raise ValueError("topk must be > 0")
                if self._top_k > E:
                    raise RuntimeError("topk must be <= number of experts")

                # (B, L, E) -> (B, L, topk)
                topk_indices = th.topk(router_logits, k=self._top_k, dim=2).indices

                # (B, L, topk) -> (B, L, E)
                expert_activations = th.zeros_like(
                    router_logits, device=self.device
                ).bool()
                expert_activations.scatter_(2, topk_indices, True)

                batch_data["activated_experts"].append(expert_activations)
                batch_data["activated_expert_indices"].append(topk_indices)
                file_tokens: list[list[str]] = output.get("tokens", [])
                batch_data["tokens"].extend(file_tokens)

            if batch_data["activated_experts"]:
                batch_data["activated_experts"] = th.cat(
                    batch_data["activated_experts"], dim=0
                )
                batch_data["activated_expert_indices"] = th.cat(
                    batch_data["activated_expert_indices"], dim=0
                )
                batch_data["top_k"] = self._top_k

                yield batch_data

    @property
    def top_k(self) -> int:
        """Get the top_k value from the dataset.

        This will load the first file to determine top_k if it hasn't been loaded yet.
        """
        if self._top_k is None:
            # Load the first file to get top_k
            file_path = os.path.join(self.dir_path, f"{self.valid_indices[0]}.pt")
            output = th.load(file_path)
            self._top_k = int(output["topk"])  # type: ignore[call-overload]
        return self._top_k  # type: ignore[return-value]


def get_activations_dataloader(
    device: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Get a DataLoader for the activations dataset.

    Args:
        device: Device to load tensors to
        batch_size: Number of files to process at once
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance
    """
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"

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
    dir_path = local_dir if os.path.isdir(local_dir) else (fallback_dir or local_dir)

    dataset = ActivationsDataset(dir_path, device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )


def get_activations_iterable_dataloader(
    device: str = "cpu",
    batch_size: int = 1,
) -> ActivationsIterableDataset:
    """Get an iterable dataset for streaming activation files.

    Args:
        device: Device to load tensors to
        batch_size: Number of files to process at once

    Returns:
        ActivationsIterableDataset instance
    """
    # Default to CPU to be CI-friendly
    device = device or "cpu"
    if device == "cuda" and not th.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU")
        device = "cpu"

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
    dir_path = local_dir if os.path.isdir(local_dir) else (fallback_dir or local_dir)

    return ActivationsIterableDataset(dir_path, device, batch_size)


if __name__ == "__main__":
    # Example usage
    dataset = get_activations_dataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Top-k: {dataset.top_k}")

    # Example of using the iterable dataset
    iterable_dataset = get_activations_iterable_dataloader(batch_size=2)
    for batch_idx, batch in enumerate(iterable_dataset):
        print(f"Batch {batch_idx}: {batch['activated_experts'].shape}")
        if batch_idx >= 2:  # Just show a few batches
            break
