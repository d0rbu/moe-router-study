from collections import defaultdict
from collections.abc import Iterator
from itertools import batched, count
import os

import torch as th
import torch.multiprocessing as mp
from tqdm import tqdm

from exp import ACTIVATION_DIRNAME, OUTPUT_DIR


class Activations:
    def __init__(
        self,
        experiment_name: str,
        device: str = "cpu",
        reshuffle: bool = False,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
        seed: int = 0,
        max_cache_size: int = 2,
    ):
        """
        Args:
            experiment_name: Name of the experiment
            device: Device to use
            reshuffle: Whether to shuffle the activations
            tokens_per_file_in_reshuffled: Number of tokens per file, only used if reshuffling
            shuffle_batch_size: How many batches to shuffle at a time
            seed: Random seed
            max_cache_size: Maximum number of files to cache
        """
        self.experiment_name = experiment_name
        self.device = device
        self.reshuffle = reshuffle
        self.tokens_per_file_in_reshuffled = tokens_per_file_in_reshuffled
        self.shuffle_batch_size = shuffle_batch_size
        self.seed = seed
        self.max_cache_size = max_cache_size

        self.activation_dir = os.path.join(
            OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME
        )

        # Load files
        self.filepaths = self.load_files(self.activation_dir, reshuffle)

        # Cache for loaded files
        self.cache = {}

    def __call__(self) -> Iterator[dict[str, th.Tensor]]:
        """Yield batches of activations.

        Yields:
            Batch of activations
        """
        # Create a queue for loading files
        queue = mp.Queue(maxsize=self.max_cache_size)
        stop_event = mp.Event()

        # Start a process to load files
        process = mp.Process(
            target=self._get_file_data,
            args=(queue, stop_event, self.filepaths, self.device),
        )
        process.start()

        # Yield batches from the queue
        try:
            while True:
                try:
                    batch = queue.get(block=True, timeout=1.0)
                    if batch is None:
                        break
                    yield batch
                except Exception:  # Use a generic exception to avoid mp.queues.Empty
                    if not process.is_alive():
                        break
        finally:
            stop_event.set()
            process.join()

    @staticmethod
    def _get_file_data(
        queue: mp.Queue,
        stop_event: mp.Event,  # type: ignore
        filepaths: list[str],
        device: str,
    ) -> None:
        """Load files into the queue.

        Args:
            queue: Queue to put data into
            stop_event: Event to signal stopping
            filepaths: List of filepaths to load
            device: Device to load data to
        """
        for filepath in filepaths:
            if stop_event.is_set():
                break

            # Load file
            data = th.load(filepath)
            data = {k: v.to(device) for k, v in data.items()}

            # Put data in queue
            queue.put(data)

        # Signal end of data
        queue.put(None)

    @classmethod
    def load_files(
        cls, activation_dir: str, reshuffle: bool = False
    ) -> list[str]:
        """Load activation files.

        Args:
            activation_dir: Directory containing activation files
            reshuffle: Whether to shuffle the activations

        Returns:
            List of filepaths
        """
        # Get filepaths
        filepaths = cls.get_activation_filepaths(activation_dir)

        # If no files and reshuffle is True, reshuffle
        if not filepaths and reshuffle:
            cls.reshuffle(activation_dir)
            filepaths = cls.get_activation_filepaths(activation_dir)

        # If still no files, raise error
        if not filepaths:
            raise FileNotFoundError(f"No activation files found in {activation_dir}")

        return filepaths

    @staticmethod
    def get_activation_filepaths(activation_dir: str) -> list[str]:
        """Get activation filepaths.

        Args:
            activation_dir: Directory containing activation files

        Returns:
            List of filepaths
        """
        # Check if directory exists
        if not os.path.exists(activation_dir):
            return []

        # Get all .pt files
        filepaths = []
        for i in count():
            filepath = os.path.join(activation_dir, f"{i}.pt")
            if not os.path.exists(filepath):
                break
            filepaths.append(filepath)

        return filepaths

    @classmethod
    def reshuffle(
        cls,
        activation_dir: str,
        tokens_per_file: int = 100_000,
        batch_size: int = 100,
        seed: int = 0,
    ) -> None:
        """Reshuffle activations.

        Args:
            activation_dir: Directory containing activation files
            tokens_per_file: Number of tokens per file
            batch_size: How many batches to shuffle at a time
            seed: Random seed
        """
        # Set random seed
        th.manual_seed(seed)

        # Get filepaths
        filepaths = cls.get_activation_filepaths(activation_dir)

        # If no files, raise error
        if not filepaths:
            raise FileNotFoundError(f"No activation files found in {activation_dir}")

        # Load all files
        all_data = defaultdict(list)
        batch_sizes = []
        for filepath in tqdm(filepaths, desc="Loading files"):
            data = th.load(filepath)
            batch_sizes.append(data["topk"].shape[0])
            for k, v in data.items():
                all_data[k].append(v)

        # Concatenate data
        all_data = {k: th.cat(v, dim=0) for k, v in all_data.items()}

        # Get total number of tokens
        total_tokens = all_data["topk"].shape[0]

        # Shuffle indices
        indices = th.randperm(total_tokens)

        # Create batches
        for i, batch_indices in enumerate(
            batched(indices, tokens_per_file)
        ):
            batch_indices = th.tensor(batch_indices)
            batch = {k: v[batch_indices] for k, v in all_data.items()}
            cls._collate_and_save_batch(batch, i, activation_dir)

    @staticmethod
    def _collate_and_save_batch(
        batch: dict[str, th.Tensor], batch_idx: int, activation_dir: str
    ) -> None:
        """Collate and save a batch.

        Args:
            batch: Batch to save
            batch_idx: Batch index
            activation_dir: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(activation_dir, exist_ok=True)

        # Save batch
        filepath = os.path.join(activation_dir, f"{batch_idx}.pt")
        th.save(batch, filepath)

    @staticmethod
    def _batch_idx_to_file_and_local_idx(
        batch_size_ranges: th.Tensor, batch_idx: int
    ) -> tuple[int, int]:
        """Convert batch index to file index and local index.

        Args:
            batch_size_ranges: Cumulative sum of batch sizes
            batch_idx: Batch index

        Returns:
            Tuple of (file_idx, local_idx)
        """
        file_idx = int(th.searchsorted(batch_size_ranges, batch_idx, side="right").item())
        local_idx = int(batch_idx - (batch_size_ranges[file_idx - 1].item() if file_idx > 0 else 0))
        return file_idx, local_idx

