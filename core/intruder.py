"""
DiskCache: A memory-efficient cache that flushes to disk when buffer is full.

This module provides a disk-backed cache for latent activations that can handle
larger-than-memory datasets by periodically flushing to disk using async I/O.
"""

from collections import defaultdict
from pathlib import Path
import time

from jaxtyping import Float, Int
from loguru import logger
from safetensors.torch import load_file, save_file
import torch as th
from torch import Tensor
import torch.multiprocessing as mp

from delphi.delphi.latents.cache import get_nonzeros_batch

location_tensor_type = Int[Tensor, "batch_sequence 3"]
activation_tensor_type = Float[Tensor, "batch_sequence"]
token_tensor_type = Int[Tensor, "batch sequence"]
latent_tensor_type = Float[Tensor, "batch sequence num_latents"]


def _disk_writer_process(write_queue: mp.Queue, done_event: mp.Event):
    """
    Background process that handles disk writes.

    Args:
        write_queue: Queue containing (output_file, data_dict) tuples to write.
        done_event: Event to signal when the process should stop.
    """
    while True:
        try:
            item = write_queue.get(timeout=0.1)
        except Exception:
            # Check if we should exit
            if done_event.is_set() and write_queue.empty():
                break
            continue

        if item is None:  # Poison pill
            break

        output_file, data_dict = item
        try:
            save_file(data_dict, output_file)
        except Exception as e:
            logger.error(f"Failed to write {output_file}: {e}")


class DiskCache:
    """
    A memory-efficient cache that stores latent locations and activations,
    flushing to disk when the buffer exceeds a specified size.

    Unlike InMemoryCache, this class periodically writes data to disk to avoid
    running out of memory on large datasets. Disk writes happen asynchronously
    in a background process to avoid blocking the main computation.
    """

    DEFAULT_CACHE_DIR = Path(".intruder_cache")
    POLL_INTERVAL_WHEN_WAITING_FOR_WRITES = 5.0
    TIMEOUT_WHEN_WAITING_FOR_WRITES = 120.0

    def __init__(
        self,
        filters: dict[str, Tensor] | None = None,
        batch_size: int = 64,
        buffer_flush_size: int = 300000,  # how many tokens before we flush to disk
        cache_dir: Path | None = None,
    ):
        """
        Initialize the DiskCache.

        Args:
            filters: Filters for selecting specific latents.
            batch_size: Size of batches for processing. Defaults to 64.
            buffer_flush_size: Number of tokens before flushing to disk.
            cache_dir: Directory to store intermediate cache files. If None,
                uses DEFAULT_CACHE_DIR.
        """
        self.filters = filters
        self.batch_size = batch_size
        self.buffer_flush_size = buffer_flush_size

        # In-memory buffers (before flushing)
        self._latent_locations_buffer: dict[str, list[location_tensor_type]] = (
            defaultdict(list)
        )
        self._latent_activations_buffer: dict[str, list[latent_tensor_type]] = (
            defaultdict(list)
        )
        self._tokens_buffer: dict[str, list[token_tensor_type]] = defaultdict(list)

        # Track current buffer size (in tokens)
        self._current_buffer_size: int = 0

        # Track flush count per hookpoint for unique filenames
        self._flush_counts: dict[str, int] = defaultdict(int)

        # Set up cache directory
        self._owns_cache_dir = cache_dir is None
        if self._owns_cache_dir:
            self._cache_dir = self.DEFAULT_CACHE_DIR
        else:
            self._cache_dir = cache_dir

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._hookpoints: set[str] = set()

        # Load metadata from cache if it exists
        if self._cache_dir.is_dir():
            # update flush counts based on filenames in the cache directory
            for subdir in self._cache_dir.iterdir():
                if not subdir.is_dir():
                    continue

                hookpoint = subdir.name
                self._hookpoints.add(hookpoint)

                max_flush_idx = -1
                for file in subdir.iterdir():
                    if file.suffix != ".safetensors":
                        continue

                    flush_idx = int(file.stem)
                    max_flush_idx = max(max_flush_idx, flush_idx)

                # Set flush count to one past the max existing file index
                self._flush_counts[hookpoint] = max_flush_idx + 1

        # Track whether save() has been called
        self._finalized = False

        # Set up async disk writer
        self._write_queue: mp.Queue = mp.Queue()
        self._done_event: mp.Event = mp.Event()
        self._writer_process: mp.Process | None = None

        logger.debug(
            f"DiskCache initialized with buffer_flush_size={buffer_flush_size:,}, "
            f"cache_dir={self._cache_dir}"
        )

    def _ensure_writer_started(self):
        """Start the background writer process if not already running."""
        if self._writer_process is None or not self._writer_process.is_alive():
            self._done_event.clear()
            self._writer_process = mp.Process(
                target=_disk_writer_process,
                args=(self._write_queue, self._done_event),
                daemon=True,
            )
            self._writer_process.start()

    def _get_hookpoint_dir(self, hookpoint: str) -> Path:
        """Get the directory for a specific hookpoint's cache files."""
        hookpoint_dir = self._cache_dir / hookpoint
        hookpoint_dir.mkdir(parents=True, exist_ok=True)
        return hookpoint_dir

    def _flush_to_disk(self):
        """Flush current buffer contents to disk asynchronously."""
        if self._current_buffer_size == 0:
            return

        logger.debug(f"Queueing {self._current_buffer_size} tokens for disk write")

        # Ensure writer is running
        self._ensure_writer_started()

        for hookpoint in self._hookpoints:
            if hookpoint not in self._latent_locations_buffer:
                continue
            if not self._latent_locations_buffer[hookpoint]:
                continue

            hookpoint_dir = self._get_hookpoint_dir(hookpoint)
            flush_idx = self._flush_counts[hookpoint]
            self._flush_counts[hookpoint] += 1

            # Concatenate and clone data to send to writer process
            concatenated_locations = th.cat(
                self._latent_locations_buffer[hookpoint], dim=0
            ).clone()
            concatenated_activations = th.cat(
                self._latent_activations_buffer[hookpoint], dim=0
            ).clone()
            concatenated_tokens = th.cat(self._tokens_buffer[hookpoint], dim=0).clone()

            # Queue the write
            output_file = str(hookpoint_dir / f"{flush_idx}.safetensors")
            data_dict = {
                "locations": concatenated_locations,
                "activations": concatenated_activations,
                "tokens": concatenated_tokens,
            }
            self._write_queue.put((output_file, data_dict))

        # Clear buffers
        self._latent_locations_buffer.clear()
        self._latent_activations_buffer.clear()
        self._tokens_buffer.clear()
        self._current_buffer_size = 0

        # Re-initialize defaultdicts
        self._latent_locations_buffer = defaultdict(list)
        self._latent_activations_buffer = defaultdict(list)
        self._tokens_buffer = defaultdict(list)

    def _get_nonzeros(
        self, latents: latent_tensor_type, module_path: str
    ) -> tuple[location_tensor_type, activation_tensor_type]:
        """
        Get the nonzero latent locations and activations.

        Args:
            latents: Input latent activations.
            module_path: Path of the module.

        Returns:
            tuple[Tensor, Tensor]: Non-zero latent locations and activations.
        """
        size = latents.shape[1] * latents.shape[0] * latents.shape[2]
        if size > th.iinfo(th.int32).max:
            nonzero_latent_locations, nonzero_latent_activations = get_nonzeros_batch(
                latents
            )
        else:
            nonzero_latent_locations = th.nonzero(latents.abs() > 1e-5)
            nonzero_latent_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero latents if no filter is provided
        if self.filters is None:
            return nonzero_latent_locations, nonzero_latent_activations

        # Return only the selected latents if a filter is provided
        selected_latents = self.filters[module_path]
        mask = th.isin(nonzero_latent_locations[:, 2], selected_latents)

        return nonzero_latent_locations[mask], nonzero_latent_activations[mask]

    def add(
        self,
        latents: latent_tensor_type,
        tokens: token_tensor_type,
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the cache.

        Args:
            latents: Latent activations.
            tokens: Input tokens.
            batch_number: Current batch number.
            module_path: Path of the module.
        """
        if self._finalized:
            raise RuntimeError("Cannot add to cache after save() has been called")

        latent_locations, latent_activations = self._get_nonzeros(latents, module_path)
        latent_locations = latent_locations.cpu()
        latent_activations = latent_activations.cpu()
        tokens = tokens.cpu()

        # Adjust batch indices
        latent_locations[:, 0] += batch_number * self.batch_size

        # Track hookpoint
        self._hookpoints.add(module_path)

        # Add to buffer
        self._latent_locations_buffer[module_path].append(latent_locations)
        self._latent_activations_buffer[module_path].append(latent_activations)
        self._tokens_buffer[module_path].append(tokens)

        # Update buffer size (count tokens)
        self._current_buffer_size += tokens.numel()

        # Flush if buffer is too large
        if self._current_buffer_size >= self.buffer_flush_size:
            self._flush_to_disk()

    def save(self):
        """
        Finalize the cache by flushing remaining data to disk.

        This method must be called before accessing data via getters.
        Waits for all pending disk writes to complete.
        """
        # Flush any remaining buffered data
        self._flush_to_disk()

        # Wait for all writes to complete
        if self._writer_process is not None and self._writer_process.is_alive():
            logger.debug(f"Waiting for {self._write_queue.qsize()} writes to complete")
            # Signal the writer to stop after processing remaining items
            self._done_event.set()
            # Send poison pill to ensure it exits
            self._write_queue.put(None)

            num_polls = (
                self.TIMEOUT_WHEN_WAITING_FOR_WRITES
                // self.POLL_INTERVAL_WHEN_WAITING_FOR_WRITES
            )

            for _ in range(num_polls):
                current_queue_size = self._write_queue.qsize()
                logger.debug(f"Queue size: {current_queue_size}")
                if current_queue_size == 0:
                    break

                time.sleep(self.POLL_INTERVAL_WHEN_WAITING_FOR_WRITES)

            self._writer_process.join(timeout=5)
            if self._writer_process.is_alive():
                logger.warning("Writer process did not terminate, forcing...")
                self._writer_process.terminate()
                self._writer_process.join(timeout=5)

        logger.debug("DiskCache finalized - all data saved to disk")
        self._finalized = True

    def __del__(self):
        """Clean up the writer process."""
        if (
            hasattr(self, "_writer_process")
            and self._writer_process is not None
            and self._writer_process.is_alive()
        ):
            self._done_event.set()
            self._write_queue.put(None)
            self._writer_process.join(timeout=5)
            if self._writer_process.is_alive():
                self._writer_process.terminate()

    @property
    def hookpoints(self) -> list[str]:
        """Return list of all hookpoints in the cache."""
        return list(self._hookpoints)

    def get_hookpoint_data(
        self, hookpoint: str
    ) -> tuple[location_tensor_type, activation_tensor_type, token_tensor_type]:
        """
        Load all data for a hookpoint from disk.

        Args:
            hookpoint: The hookpoint to load data for.

        Returns:
            Tuple of (locations, activations, tokens) tensors.
        """
        if not self._finalized:
            raise RuntimeError("Must call save() before accessing data")

        hookpoint_dir = self._get_hookpoint_dir(hookpoint)

        locations_list = []
        activations_list = []
        tokens_list = []

        # Load all batch files for this hookpoint (sorted by flush index)
        batch_files = sorted(
            hookpoint_dir.glob("*.safetensors"),
            key=lambda p: int(p.stem),
        )

        for batch_file in batch_files:
            data = load_file(batch_file)
            locations_list.append(data["locations"])
            activations_list.append(data["activations"])
            tokens_list.append(data["tokens"])

        if not locations_list:
            raise KeyError(f"No data found for hookpoint: {hookpoint}")

        locations = th.cat(locations_list, dim=0)
        activations = th.cat(activations_list, dim=0)
        tokens = th.cat(tokens_list, dim=0)

        return locations, activations, tokens
