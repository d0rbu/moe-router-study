from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched, count, islice, pairwise
import os
import threading
import time

from loguru import logger
import torch as th
import torch.distributed as dist
from tqdm import tqdm

from core.device import DeviceType, get_backend, get_distributed_backend
from core.logging import init_distributed_logging
from core.memory import clear_memory
from core.type import assert_type
from exp import ACTIVATION_DIRNAME, OUTPUT_DIR
from exp.get_activations import ActivationKeys
from exp.training import get_experiment_name


def broadcast_variable_length_list[T](
    list_fn: Callable[..., list[T]],
    src: int = 0,
    args: tuple = (),
    kwargs: dict | None = None,
) -> list[T]:
    if kwargs is None:
        kwargs = {}

    # Broadcast the number of items first
    num_items = [None]
    if dist.get_rank() == 0:
        items = list_fn(*args, **kwargs)
        num_items[0] = len(items)

    dist.broadcast_object_list(num_items, src=src)
    num_items = num_items[0]
    assert isinstance(num_items, int), "num_items should be an integer after broadcast"

    logger.trace(f"Rank {src} broadcasted that there are {num_items} items")

    if num_items == 0:
        return []

    # Now broadcast the actual items
    if dist.get_rank() != 0:
        items = [None] * num_items

    dist.broadcast_object_list(items, src=src)
    logger.trace(f"Rank {src} broadcasted {num_items} items")

    # At this point, items contains the actual values from rank 0
    return assert_type(items, list)


class Activations:
    def __init__(
        self,
        device: str = "cpu",
        activation_filepaths: list[str] | None = None,
        max_cache_size: int = 2,
    ):
        """
        Args:
            device: Device to return the activations on
            activation_filepaths: List of activation filepaths
            max_cache_size: Maximum number of file data entries to cache when iterating
        """
        self.device = device
        self.activation_filepaths = activation_filepaths
        self.max_cache_size = max_cache_size
        self._total_tokens = None

        # Shared file cache and position tracking
        self._file_cache = {}  # {file_index: file_data}
        self._file_positions = {}  # {thread_id: current_file_index}
        self._positions_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache_update_event = threading.Event()

        # Shared worker process for file loading
        self.file_worker = None
        self._worker_lock = threading.Lock()

    @classmethod
    def load(
        cls,
        experiment_name: str,
        device: str = "cpu",
        tokens_per_file_in_reshuffled: int = 10_000,
        shuffle_batch_size: int = 10,
        seed: int = 0,
        max_cache_size: int = 2,
        num_workers: int = 8,
        debug: bool = False,
        device_type: DeviceType = "cuda",
    ) -> "Activations":
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "PyTorch distributed training is not initialized. "
                "Ensure that PyTorch distributed training is initialized with "
                "`dist.init_process_group`. You can use `load_activations_and_init_dist()` "
                "to have this done automatically from SLURM environment variables. "
                "This function handles both single and multi-process execution and ensures "
                "proper distributed training setup."
            )

        activation_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

        cls.device = device

        logger.trace(f"Loading or reshuffling activations from {activation_dir}")
        cls.activation_filepaths = cls.load_files(
            activation_dir=activation_dir,
            seed=seed,
            tokens_per_file_in_reshuffled=tokens_per_file_in_reshuffled,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
            device_type=device_type,
        )

        return cls(
            device=device,
            activation_filepaths=cls.activation_filepaths,
            max_cache_size=max_cache_size,
        )

    def __len__(self) -> int:
        if self._total_tokens is not None:
            return self._total_tokens

        num_tokens = th.zeros(1, dtype=th.int32)
        local_activation_filepath_iterator = islice(
            self.activation_filepaths,
            dist.get_rank(),  # start
            len(self.activation_filepaths),  # stop
            dist.get_world_size(),  # step
        )

        for filepath in local_activation_filepath_iterator:
            activations = th.load(filepath, weights_only=False)
            num_tokens += activations[ActivationKeys.MLP_OUTPUT].shape[0]

        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
        self._total_tokens = num_tokens.item()

        return self._total_tokens

    def _file_worker_loop(self):
        """Background worker that maintains the file cache based on thread positions."""
        logger.debug("File worker started")

        while True:
            # Wait for a signal that positions have changed
            self._cache_update_event.wait()
            self._cache_update_event.clear()

            # Compute desired file set based on current positions
            with self._positions_lock:
                if not self._file_positions:
                    # No active threads, nothing to do
                    logger.trace("No active threads, worker idling")
                    continue

                positions = list(self._file_positions.values())

            # Compute fair cache distribution
            num_threads = len(positions)
            files_per_thread = self.max_cache_size // num_threads

            if files_per_thread == 0:
                logger.warning(
                    f"Cache size {self.max_cache_size} too small for {num_threads} threads"
                )
                files_per_thread = 1

            desired_files = set()
            allocated_files = set()

            # Allocate files fairly among threads
            for position in sorted(positions):
                thread_files = []
                file_idx = position

                while len(thread_files) < files_per_thread:
                    if file_idx >= len(self.activation_filepaths):
                        break
                    if file_idx not in allocated_files:
                        thread_files.append(file_idx)
                        allocated_files.add(file_idx)
                    file_idx += 1

                desired_files.update(thread_files)

                if len(allocated_files) >= self.max_cache_size:
                    break

            # Determine files to load and evict
            with self._cache_lock:
                current_files = set(self._file_cache.keys())
                files_to_load = desired_files - current_files
                files_to_evict = current_files - desired_files

                logger.trace(
                    f"Worker: current={sorted(current_files)}, "
                    f"desired={sorted(desired_files)}, "
                    f"load={sorted(files_to_load)}, "
                    f"evict={sorted(files_to_evict)}"
                )

                # Evict files no longer needed
                for file_idx in files_to_evict:
                    logger.trace(f"Evicting file {file_idx}")
                    del self._file_cache[file_idx]

                # Load new files
                for file_idx in sorted(files_to_load):
                    if file_idx < len(self.activation_filepaths):
                        filepath = self.activation_filepaths[file_idx]
                        logger.debug(f"Loading file {file_idx}: {filepath}")
                        file_data = th.load(filepath, weights_only=False)
                        self._file_cache[file_idx] = file_data
                        logger.debug(
                            f"Loaded file {file_idx}, cache size={len(self._file_cache)}"
                        )

    def _load_files_sequentially(self) -> Generator[dict, None, None]:
        """Generator that loads files sequentially, coordinating with the shared worker."""
        # Acquire a thread ID
        with self._positions_lock:
            # Find first available integer ID
            thread_id = 0
            while thread_id in self._file_positions:
                thread_id += 1

            self._file_positions[thread_id] = 0
            logger.debug(f"Thread {thread_id} acquired, starting at file 0")

        # Signal worker to load initial files
        self._cache_update_event.set()

        try:
            for file_idx in range(len(self.activation_filepaths)):
                # Wait for file to be in cache
                while True:
                    with self._cache_lock:
                        if file_idx in self._file_cache:
                            file_data = self._file_cache[file_idx]
                            logger.trace(
                                f"Thread {thread_id} accessing file {file_idx} from cache"
                            )
                            break

                    # File not ready, wait a bit
                    logger.trace(f"Thread {thread_id} waiting for file {file_idx}")
                    time.sleep(0.01)

                # Yield the file data
                yield file_data

                # Update position and signal worker
                with self._positions_lock:
                    if thread_id in self._file_positions:
                        self._file_positions[thread_id] = file_idx + 1
                        logger.trace(
                            f"Thread {thread_id} moved to position {file_idx + 1}"
                        )

                self._cache_update_event.set()

        finally:
            # Clean up thread tracking
            with self._positions_lock:
                if thread_id in self._file_positions:
                    del self._file_positions[thread_id]
                    logger.debug(f"Thread {thread_id} released")

            # Signal worker to update cache
            self._cache_update_event.set()

    def __call__(
        self, batch_size: int = 4096, start_idx: int = 0, max_samples: int = 0
    ) -> Generator[dict, None, None]:
        assert max_samples >= 0, f"max_samples must be non-negative, got {max_samples}"

        # Ensure worker is started (lazy initialization)
        with self._worker_lock:
            if self.file_worker is None:
                self.file_worker = threading.Thread(
                    target=self._file_worker_loop, daemon=True
                )
                self.file_worker.start()
                logger.debug("Started shared file worker thread")

        # Track samples processed for max_samples limit
        samples_processed = 0

        # Use the shared file loading mechanism
        file_iterator = self._load_files_sequentially()
        skipped_start = False

        try:
            current_data = next(file_iterator)

            logger.debug(
                f"Loaded data with shape {current_data[ActivationKeys.MLP_OUTPUT].shape}"
            )

            current_local_idx = 0
            current_data_size = current_data[ActivationKeys.MLP_OUTPUT].shape[0]

            current_batch = {}
            remaining_batch_size = start_idx

            for _batch_idx in count():
                while current_data_size - current_local_idx <= remaining_batch_size:
                    for key, value in current_data.items():
                        match value:
                            case th.Tensor():
                                if key in current_batch:
                                    current_batch[key] = th.cat(
                                        [
                                            current_batch[key],
                                            value[
                                                current_local_idx : current_local_idx
                                                + batch_size
                                            ].to(self.device),
                                        ],
                                        dim=0,
                                    )
                                else:
                                    current_batch[key] = value[
                                        current_local_idx : current_local_idx
                                        + batch_size
                                    ].to(self.device)
                            case list():
                                if key in current_batch:
                                    assert current_batch[key] == value, (
                                        f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                    )
                                else:
                                    current_batch[key] = value
                            case _:
                                if key in current_batch:
                                    assert current_batch[key] == value, (
                                        f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                    )
                                else:
                                    current_batch[key] = value

                    remaining_batch_size -= current_data_size - current_local_idx

                    try:
                        current_data = next(file_iterator)
                    except StopIteration:
                        logger.debug("No more data to load, iteration complete")
                        return

                    current_local_idx = 0
                    current_data_size = current_data[ActivationKeys.MLP_OUTPUT].shape[0]
                else:
                    for key, value in current_data.items():
                        match value:
                            case th.Tensor():
                                if key in current_batch:
                                    current_batch[key] = th.cat(
                                        [
                                            current_batch[key],
                                            value[
                                                current_local_idx : current_local_idx
                                                + remaining_batch_size
                                            ].to(self.device),
                                        ],
                                        dim=0,
                                    )
                                else:
                                    current_batch[key] = value[
                                        current_local_idx : current_local_idx
                                        + remaining_batch_size
                                    ].to(self.device)
                            case list():
                                if key in current_batch:
                                    assert current_batch[key] == value, (
                                        f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                    )
                                else:
                                    current_batch[key] = value
                            case _:
                                if key in current_batch:
                                    assert current_batch[key] == value, (
                                        f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                    )
                                else:
                                    current_batch[key] = value

                    assert len(current_batch) > 0, "Current batch is empty"

                    if skipped_start:
                        # Check max_samples limit before yielding
                        if max_samples > 0:
                            batch_size_actual = current_batch[
                                ActivationKeys.ROUTER_LOGITS
                            ].shape[0]
                            if samples_processed + batch_size_actual > max_samples:
                                batch_size_actual = max_samples - samples_processed
                                if batch_size_actual > 0:
                                    # Truncate current_batch in place
                                    for key, value in current_batch.items():
                                        if isinstance(value, th.Tensor):
                                            current_batch[key] = value[
                                                :batch_size_actual
                                            ]

                            samples_processed += batch_size_actual
                            yield current_batch
                            if samples_processed >= max_samples:
                                break
                        else:
                            yield current_batch
                    else:
                        skipped_start = True

                    current_batch = {}
                    current_local_idx += remaining_batch_size
                    remaining_batch_size = batch_size

                    clear_memory()

        except GeneratorExit:
            # Handle generator close() - clean up resources
            logger.debug("GeneratorExit received, closing file iterator")
            file_iterator.close()
            logger.trace("File iterator closed")
            clear_memory()
            raise  # Re-raise GeneratorExit to properly close the generator

    def __iter__(self) -> Generator[dict, None, None]:
        return self()

    @classmethod
    def load_files(
        cls,
        activation_dir: str,
        seed: int = 0,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 10,
        debug: bool = False,
        num_workers: int = 8,
        device_type: DeviceType = "cuda",
    ) -> list[str]:
        shuffle_dirname = (
            f"reshuffled-seed={seed}-tokens_per_file={tokens_per_file_in_reshuffled}"
        )
        activation_files_dir = os.path.join(activation_dir, shuffle_dirname)
        if dist.get_rank() == 0:
            os.makedirs(activation_files_dir, exist_ok=True)
        dist.barrier()

        activation_filepaths = cls.get_activation_filepaths(
            activation_files_dir, debug=debug
        )
        logger.trace(f"Found shuffled activation files {activation_filepaths}")

        if activation_filepaths:
            return activation_filepaths

        # if we are here then there are no activation files
        logger.info(
            f"Reshuffling activations from {activation_dir} to {activation_files_dir}"
        )

        new_activation_filepaths = cls.reshuffle(
            activation_dir=activation_dir,
            output_dir=activation_files_dir,
            tokens_per_file_in_reshuffled=tokens_per_file_in_reshuffled,
            seed=seed,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
            device_type=device_type,
        )

        return new_activation_filepaths

    @classmethod
    def get_activation_filepaths(
        cls, activation_dir: str, debug: bool = False
    ) -> list[str]:
        all_activation_filenames = {
            filename
            for filename in os.listdir(activation_dir)
            if filename.endswith(".pt")
        }

        logger.trace(f"Found {len(all_activation_filenames)} activation files")

        activation_indices = {
            int(filename.split(".")[0]) for filename in all_activation_filenames
        }

        logger.trace(f"Found {len(activation_indices)} activation indices")

        if len(activation_indices) == 0:
            return []

        max_contiguous_activation_index = max(activation_indices)
        for prev_index, next_index in pairwise(sorted(activation_indices)):
            if next_index - prev_index > 1:
                max_contiguous_activation_index = prev_index
                break

        logger.trace(
            f"Max contiguous activation index: {max_contiguous_activation_index}"
        )

        contiguous_activation_filepaths = [
            os.path.join(activation_dir, f"{i}.pt")
            for i in range(max_contiguous_activation_index + 1)
        ]

        if not debug:
            return contiguous_activation_filepaths

        truncated_contiguous_activation_filepaths = contiguous_activation_filepaths[
            : cls.NUM_DEBUG_FILES
        ]

        return truncated_contiguous_activation_filepaths

    @staticmethod
    def load_files_async(filepaths: list[str]) -> list[dict]:
        with ThreadPoolExecutor() as executor:
            load_unsafe = partial(th.load, weights_only=False)
            futures = [executor.submit(load_unsafe, filepath) for filepath in filepaths]
            results = [future.result() for future in futures]
        return results

    NUM_DEBUG_FILES = 32

    @classmethod
    def reshuffle(
        cls,
        activation_dir: str,
        output_dir: str,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 10,
        seed: int = 0,
        debug: bool = False,
        num_workers: int = 8,
        cpu_only: bool = False,
        device_type: DeviceType = "cuda",
    ) -> list[str]:
        # Get backend for device operations
        backend = get_backend(device_type)

        # Set rank and world_size based on cpu_only mode
        if cpu_only:
            rank = 0
            world_size = 1
            activation_filepaths = cls.get_activation_filepaths(activation_dir, debug)
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            activation_filepaths = broadcast_variable_length_list(
                cls.get_activation_filepaths,
                args=(activation_dir, debug),
                src=0,
            )

        assert len(activation_filepaths) > 0, "No activation files found :("

        new_activation_filepaths = []
        all_batch_sizes = th.zeros(len(activation_filepaths), dtype=th.int32)

        activation_filepath_limit = len(activation_filepaths)
        if debug:
            logger.debug(f"Debug mode, only loading first {cls.NUM_DEBUG_FILES} files")
            activation_filepath_limit = min(
                activation_filepath_limit, cls.NUM_DEBUG_FILES
            )

        local_activation_filepaths = activation_filepaths[
            rank:activation_filepath_limit:world_size
        ]

        # start threadpool to load the files
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            load_unsafe = partial(th.load, weights_only=False)
            futures = deque(
                executor.submit(load_unsafe, filepath)
                for filepath in local_activation_filepaths
            )
            for future_idx in tqdm(
                range(rank, activation_filepath_limit, world_size),
                total=len(futures),
                desc="Loading batch sizes",
                leave=False,
                position=rank,
            ):
                future = futures.popleft()
                data = future.result()
                all_batch_sizes[future_idx] = data[ActivationKeys.MLP_OUTPUT].shape[0]
                del data
                del future

        if not cpu_only:
            dist.all_reduce(all_batch_sizes, op=dist.ReduceOp.SUM)

        current_batch = defaultdict(list)
        current_batch_idx = 0
        total_tokens = 0
        num_batch_tokens = 0

        activation_file_batches = list(
            batched(
                zip(activation_filepaths, all_batch_sizes, strict=False),
                shuffle_batch_size,
            )
        )
        local_activation_file_batches = activation_file_batches[
            rank:activation_filepath_limit:world_size
        ]

        for shuffle_batch_idx, shuffle_batch in tqdm(
            enumerate(local_activation_file_batches),
            desc=f"Rank {rank}" if not cpu_only else "Processing batches",
            total=len(local_activation_file_batches),
            leave=False,
            position=rank * 2,
        ):
            current_seed = (
                seed * len(local_activation_file_batches) * world_size
                + shuffle_batch_idx * world_size
                + rank
            )
            th.manual_seed(current_seed)
            if not cpu_only:
                backend.manual_seed_all(current_seed)

            filepaths, batch_sizes = zip(*shuffle_batch, strict=True)

            file_data = cls.load_files_async(filepaths)

            batch_sizes = th.stack(batch_sizes, dim=0)
            batch_size_ranges = th.cumsum(batch_sizes, dim=0)
            total_size = batch_size_ranges[-1].item()

            expected_combinations = {
                (file_idx, local_idx)
                for file_idx in range(len(filepaths))
                for local_idx in range(batch_sizes[file_idx].item())
            }
            processed_combinations = set()

            batch_shuffled_indices = th.randperm(total_size)

            for batch_idx in tqdm(
                batch_shuffled_indices,
                desc=f"Shuffle batch {shuffle_batch_idx}",
                total=total_size,
                leave=False,
                position=rank * 2 + 1,
            ):
                file_idx, local_idx = Activations._batch_idx_to_file_and_local_idx(
                    batch_size_ranges, batch_idx
                )

                combination = (file_idx, local_idx)

                assert combination not in processed_combinations, (
                    f"Duplicate combination {combination} in shuffle batch {shuffle_batch_idx}"
                )
                assert combination in expected_combinations, (
                    f"Invalid combination {combination} in shuffle batch {shuffle_batch_idx}"
                )

                processed_combinations.add(combination)
                batch_size = batch_sizes[file_idx]
                data = file_data[file_idx]
                data_to_copy = data.copy()
                # don't copy over the tokens since we are mixing them up anyway
                del data_to_copy["tokens"]

                for raw_key, value in data_to_copy.items():
                    # in case raw_key is an enum
                    key = str(raw_key)

                    if isinstance(value, th.Tensor | list) and len(value) == batch_size:
                        current_batch[key].append(value[local_idx])
                        continue

                    if key in current_batch:
                        assert current_batch[key] == value, (
                            f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                        )
                    else:
                        current_batch[key] = value

                num_batch_tokens += 1

                if num_batch_tokens >= tokens_per_file_in_reshuffled:
                    output_filepath = os.path.join(
                        output_dir, f"{rank}_{current_batch_idx}.pt-temp"
                    )
                    new_activation_filepaths.append(
                        cls._collate_and_save_batch(current_batch, output_filepath)
                    )

                    current_batch = defaultdict(list)
                    total_tokens += num_batch_tokens
                    num_batch_tokens = 0
                    current_batch_idx += 1

            unprocessed_combinations = expected_combinations - processed_combinations
            extra_combinations = processed_combinations - expected_combinations
            assert len(unprocessed_combinations) == 0, (
                f"Missing combinations in shuffle batch {shuffle_batch_idx}: {unprocessed_combinations}"
            )
            assert len(extra_combinations) == 0, (
                f"Extra combinations in shuffle batch {shuffle_batch_idx}: {extra_combinations}"
            )

        total_tokens += num_batch_tokens
        total_tokens = th.tensor(total_tokens, dtype=th.int32)

        if cpu_only:
            remaining_stacked_batches = [cls._stack_batch_for_gather(current_batch)]
            logger.debug(f"Total tokens: {total_tokens.item()}")
        else:
            dist.reduce(total_tokens, dst=0, op=dist.ReduceOp.SUM)

            if rank == 0:
                remaining_stacked_batches = [None] * world_size
                logger.debug(f"Total tokens: {total_tokens.item()}")
            else:
                remaining_stacked_batches = None

            # Stack lists of tensors before gathering to improve performance
            stacked_current_batch = cls._stack_batch_for_gather(current_batch)
            dist.gather_object(stacked_current_batch, remaining_stacked_batches, dst=0)

        if rank == 0:
            assert remaining_stacked_batches is not None
            logger.debug(f"Gathered {len(remaining_stacked_batches)} batches")

            # Unstack the gathered batches back to original format
            remaining_batches = [
                cls._unstack_batch_after_gather(batch)
                for batch in remaining_stacked_batches
                if batch
            ]

            extra_activation_filepaths = []
            if len(remaining_batches) > 0:
                concatenated_batch = remaining_batches[0]
                for batch in remaining_batches[1:]:
                    for key, value in batch.items():
                        if isinstance(value, list):
                            concatenated_batch[key].extend(value)
                        else:
                            concatenated_batch[key] = value

                total_extra_tokens = len(concatenated_batch[ActivationKeys.MLP_OUTPUT])
                num_extra_batches = total_extra_tokens // tokens_per_file_in_reshuffled
                tokens_skipped = total_extra_tokens % tokens_per_file_in_reshuffled

                logger.info(f"Skipping {tokens_skipped} tokens for even batching")

                for extra_batch_idx in tqdm(
                    range(num_extra_batches),
                    desc="Getting extra activation filepaths",
                    total=num_extra_batches,
                    leave=False,
                ):
                    start_idx = extra_batch_idx * tokens_per_file_in_reshuffled
                    end_idx = start_idx + tokens_per_file_in_reshuffled
                    extra_batch = {}
                    for key, value in concatenated_batch.items():
                        if isinstance(value, list):
                            extra_batch[key] = value[start_idx:end_idx]
                        else:
                            extra_batch[key] = value

                    output_filepath = os.path.join(
                        output_dir,
                        f"{rank}_{current_batch_idx + extra_batch_idx}.pt-temp",
                    )
                    extra_activation_filepaths.append(
                        cls._collate_and_save_batch(extra_batch, output_filepath)
                    )

            new_activation_filepaths.extend(extra_activation_filepaths)
            reshuffled_indices = th.randperm(
                len(new_activation_filepaths),
                generator=th.Generator().manual_seed(seed),
            )

            for new_idx, filepath in tqdm(
                zip(reshuffled_indices, new_activation_filepaths, strict=True),
                desc="Reshuffling output files",
                leave=False,
                total=len(new_activation_filepaths),
            ):
                os.rename(filepath, os.path.join(output_dir, f"{new_idx}.pt"))

            reshuffled_activation_filenames = [
                f"{i}.pt" for i in range(len(new_activation_filepaths))
            ]
            renamed_activation_filepaths = [
                os.path.join(output_dir, filename)
                for filename in reshuffled_activation_filenames
            ]
        else:
            renamed_activation_filepaths = []

        if not cpu_only:
            renamed_activation_filepaths = broadcast_variable_length_list(
                lambda: renamed_activation_filepaths,
                src=0,
            )

        return renamed_activation_filepaths

    @staticmethod
    def _stack_batch_for_gather(batch: dict) -> dict:
        """Stack lists of tensors into single tensors for efficient gathering."""
        stacked_batch = {}
        for key, value in batch.items():
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], th.Tensor)
            ):
                stacked_batch[key] = th.stack(value, dim=0)

        return stacked_batch

    @staticmethod
    def _unstack_batch_after_gather(batch: dict) -> dict:
        """Unstack tensors back into lists of tensors after gathering."""
        unstacked_batch = {}
        for key, value in batch.items():
            if isinstance(value, th.Tensor) and value.ndim > 0:
                # Unstack tensor back into list of tensors
                unstacked_batch[key] = list(value)
            else:
                # Keep other values as-is
                unstacked_batch[key] = value
        return unstacked_batch

    @staticmethod
    def _collate_and_save_batch(batch: dict, output_filepath: str) -> str:
        for key, value in batch.items():
            if isinstance(value, list):
                if len(value) == 0:
                    del batch[key]
                    continue

                if not isinstance(value[0], th.Tensor):
                    continue

                batch[key] = th.stack(value, dim=0)

        th.save(batch, output_filepath)

        return output_filepath

    @staticmethod
    def _batch_idx_to_file_and_local_idx(
        batch_size_ranges: th.Tensor, batch_idx: int | th.Tensor
    ) -> tuple[int, int]:
        if isinstance(batch_idx, th.Tensor):
            assert batch_idx.ndim == 0, "Batch index must be a scalar"
            batch_idx = batch_idx.item()

        file_idx = th.searchsorted(batch_size_ranges, batch_idx, side="right").item()

        file_start_idx = batch_size_ranges[file_idx - 1].item() if file_idx > 0 else 0

        local_idx = batch_idx - file_start_idx
        return file_idx, local_idx


def load_activations_and_init_dist(
    model_name: str,
    dataset_name: str,
    tokens_per_file: int,
    reshuffled_tokens_per_file: int,
    submodule_names: list[str],
    context_length: int,
    seed: int = 0,
    num_workers: int = 8,
    debug: bool = False,
    device_type: DeviceType = "cuda",
) -> tuple[
    Activations,
    dict[str, int],
    dist.ProcessGroup | None,
    list[dist.ProcessGroup] | None,
]:
    """
    Load activations and initialize the distributed process group.

    Returns:
        activations: Activations object
        activation_dims: Dimensionality of the activations for each submodule as a dictionary
        gpu_process_group: General GPU process group (or None if GPU not available)
        gpu_process_groups: List of GPU-specific process groups, one per device (or None if GPU not available)
    """
    activations_experiment_name = get_experiment_name(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        context_length=context_length,
    )
    logger.debug(f"Loading from experiment {activations_experiment_name}")

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    gloo_port = int(os.environ.get("MASTER_PORT", 10000))

    logger.debug("Initializing distributed process group")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    logger.info(f"Rank {rank} initialized gloo group")

    backend = get_backend(device_type)
    logger.debug(f"Backend: {backend}")
    logger.debug(f"Backend is available: {backend.is_available()}")
    if backend.is_available():
        num_devices = backend.device_count()
        logger.debug(f"Number of devices: {num_devices}")

        nccl_port = gloo_port + 1
        os.environ["MASTER_PORT"] = str(nccl_port)

        backend_name = get_distributed_backend(device_type)
        gpu_process_group = dist.new_group(
            ranks=list(range(world_size)), backend=backend_name
        )
        logger.info(f"Rank {rank} initialized {backend_name} group")

        gpu_process_groups = [
            dist.new_group(ranks=list(range(world_size)), backend=backend_name)
            for _ in range(num_devices)
        ]
    else:
        gpu_process_group = None
        gpu_process_groups = None

    init_distributed_logging()

    logger.debug(f"Initializing activations with seed {seed}")
    activations = Activations.load(
        experiment_name=activations_experiment_name,
        tokens_per_file_in_reshuffled=reshuffled_tokens_per_file,
        seed=seed,
        num_workers=num_workers,
        debug=debug,
        device_type=device_type,
    )

    # load a batch of activations to get the dimension
    data_iterable = activations(batch_size=1)

    logger.debug(f"Activation filepaths count: {len(activations.activation_filepaths)}")
    if activations.activation_filepaths:
        logger.debug(f"First filepath: {activations.activation_filepaths[0]}")

        first_file = activations.activation_filepaths[0]
        if os.path.exists(first_file):
            try:
                sample_data = th.load(first_file, weights_only=False)
                logger.debug(f"Sample data keys: {list(sample_data.keys())}")
                if ActivationKeys.MLP_OUTPUT in sample_data:
                    logger.debug(
                        f"MLP output shape: {sample_data[ActivationKeys.MLP_OUTPUT].shape}"
                    )
                else:
                    logger.error(
                        f"{ActivationKeys.MLP_OUTPUT} key not found in activation data"
                    )
            except Exception as e:
                logger.error(f"Error loading first file {first_file}: {e}")
        else:
            logger.error(f"First file does not exist: {first_file}")

    try:
        activation = next(data_iterable)
    except StopIteration as e:
        logger.error("Data iterator is empty - no activations found")

        raise RuntimeError(
            "No activation data found. Check that activation files are properly created and contain data."
        ) from e

    logger.trace(
        f"Activation: {', '.join(f'{key}: {value.shape}' for key, value in activation.items() if isinstance(value, th.Tensor))}"
    )
    activation_dims = {
        submodule_name: th.prod(th.tensor(activation[submodule_name].shape[2:])).item()
        for submodule_name in submodule_names
    }

    # for router logits, we flatten out the layer dimension
    if ActivationKeys.ROUTER_LOGITS in submodule_names:
        activation_dims[ActivationKeys.ROUTER_LOGITS] *= activation[
            ActivationKeys.ROUTER_LOGITS
        ].shape[1]
    logger.debug(f"Activation dims: {activation_dims}")

    # clean up the background worker and queue
    data_iterable.close()

    return activations, activation_dims, gpu_process_group, gpu_process_groups
