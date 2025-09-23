import asyncio
from collections import defaultdict, deque
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
import gc
from itertools import batched, count, islice, pairwise
import os

from loguru import logger
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from core.logging import init_distributed_logging
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

    num_items = [None]

    if dist.get_rank() == 0:
        items = list_fn(*args, **kwargs)
        num_items[0] = len(items)

    dist.broadcast_object_list(num_items, src=src)
    num_items = num_items[0]

    logger.trace(f"Rank {src} broadcasted that there are {num_items} items")

    if num_items == 0:
        return []

    if dist.get_rank() != 0:
        items = [None] * num_items

    dist.broadcast_object_list(items, src=src)
    logger.trace(f"Rank {src} broadcasted {num_items} items")

    return items


class Activations:
    def __init__(
        self,
        experiment_name: str,
        device: str = "cpu",
        tokens_per_file_in_reshuffled: int = 10_000,
        shuffle_batch_size: int = 10,
        seed: int = 0,
        max_cache_size: int = 2,
        num_workers: int = 8,
        debug: bool = False,
    ):
        """
        Args:
            experiment_name: Name of the experiment
            device: Device to return the activations on
            tokens_per_file_in_reshuffled: Number of tokens per file, only used if reshuffling
            shuffle_batch_size: How many batches to shuffle at a time
            seed: Seed for the random number generator
            max_cache_size: Maximum number of file data entries to cache when iterating
            num_workers: Number of workers to use for loading the files
            debug: Whether to run in debug mode
        """
        activation_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

        self.device = device

        logger.trace(f"Loading or reshuffling activations from {activation_dir}")
        self.activation_filepaths = self.load_files(
            activation_dir=activation_dir,
            seed=seed,
            tokens_per_file_in_reshuffled=tokens_per_file_in_reshuffled,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
        )

        self.max_cache_size = max_cache_size
        self._total_tokens = None

    def __len__(self) -> int:
        if self._total_tokens is not None:
            return self._total_tokens

        num_tokens = th.zeros(0, dtype=th.int32)
        local_activation_filepath_iterator = islice(
            self.activation_filepaths,
            dist.get_rank(),  # start
            len(self.activation_filepaths),  # stop
            dist.get_world_size(),  # step
        )

        for filepath in local_activation_filepath_iterator:
            activations = th.load(filepath)
            num_tokens += activations[ActivationKeys.MLP_OUTPUT].shape[0]

        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)
        self._total_tokens = num_tokens.item()

        return self._total_tokens

    # worker to fetch data from disk
    def _get_file_data(self, cached_file_data: mp.Queue):
        for activation_filepath in self.activation_filepaths:
            file_data = th.load(activation_filepath)
            cached_file_data.put(file_data, block=True)
        cached_file_data.put(None)

    def __call__(self, batch_size: int = 4096) -> Generator[dict, None, None]:
        # cache of file data to come
        cached_file_data = mp.Queue(maxsize=self.max_cache_size)

        # create worker process to get file data
        worker_process = mp.Process(
            target=self._get_file_data, args=(cached_file_data,)
        )
        worker_process.start()

        current_data = cached_file_data.get(block=True)
        current_local_idx = 0
        current_data_size = len(current_data["tokens"])

        current_batch = {}
        remaining_batch_size = batch_size

        for _batch_idx in count():
            while current_data_size - current_local_idx < remaining_batch_size:
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
                                    current_local_idx : current_local_idx + batch_size
                                ].to(self.device)
                        case list():
                            pass
                        case _:
                            if key in current_batch:
                                assert current_batch[key] == value, (
                                    f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                )
                            else:
                                current_batch[key] = value

                remaining_batch_size -= current_data_size - current_local_idx
                current_data = cached_file_data.get(block=True)

                if current_data is None:
                    worker_process.join()
                    cached_file_data.close()
                    return

                current_local_idx = 0
                current_data_size = len(current_data["tokens"])
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
                                            + batch_size
                                        ].to(self.device),
                                    ],
                                    dim=0,
                                )
                            else:
                                current_batch[key] = value[
                                    current_local_idx : current_local_idx + batch_size
                                ].to(self.device)
                        case list():
                            pass
                        case _:
                            if key in current_batch:
                                assert current_batch[key] == value, (
                                    f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                )
                            else:
                                current_batch[key] = value

                stop = yield current_batch

                if stop is not None:
                    # this is the stop signal, so we stop the process and queue
                    worker_process.terminate()
                    cached_file_data.close()
                    return

                current_batch = {}
                current_local_idx += remaining_batch_size
                remaining_batch_size = batch_size

                th.cuda.empty_cache()
                gc.collect()

    def __iter__(self) -> Generator[dict, None, None]:
        return self()

    def load_files(
        self,
        activation_dir: str,
        seed: int = 0,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
        debug: bool = False,
        num_workers: int = 8,
    ) -> list[str]:
        shuffle_dirname = (
            f"reshuffled-seed={seed}-tokens_per_file={tokens_per_file_in_reshuffled}"
        )
        activation_files_dir = os.path.join(activation_dir, shuffle_dirname)
        if dist.get_rank() == 0:
            os.makedirs(activation_files_dir, exist_ok=True)
        dist.barrier()

        activation_filepaths = self.get_activation_filepaths(activation_files_dir)
        logger.trace(f"Found shuffled activation files {activation_filepaths}")

        if activation_filepaths:
            return activation_filepaths

        # if we are here then there are no activation files
        logger.info(
            f"Reshuffling activations from {activation_dir} to {activation_files_dir}"
        )

        new_activation_filepaths = self.reshuffle(
            activation_dir=activation_dir,
            output_dir=activation_files_dir,
            tokens_per_file_in_reshuffled=tokens_per_file_in_reshuffled,
            seed=seed,
            shuffle_batch_size=shuffle_batch_size,
            debug=debug,
            num_workers=num_workers,
        )

        return new_activation_filepaths

    @staticmethod
    def get_activation_filepaths(activation_dir: str) -> list[str]:
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
        return contiguous_activation_filepaths

    @staticmethod
    async def load_files_async(filepaths: list[str]) -> list[dict]:
        return await asyncio.gather(
            *[asyncio.to_thread(th.load, filepath) for filepath in filepaths]
        )

    NUM_DEBUG_FILES = 2

    def reshuffle(
        self,
        activation_dir: str,
        output_dir: str,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
        seed: int = 0,
        debug: bool = False,
        num_workers: int = 8,
    ) -> list[str]:
        self._total_tokens = None

        activation_filepaths = broadcast_variable_length_list(
            self.get_activation_filepaths,
            args=(activation_dir,),
            src=0,
        )

        assert len(activation_filepaths) > 0, "No activation files found :("

        new_activation_filepaths = []
        all_batch_sizes = th.zeros(len(activation_filepaths), dtype=th.int32)

        activation_filepath_limit = len(activation_filepaths)
        if debug:
            logger.info(
                f"Debug mode, only loading first {self.NUM_DEBUG_FILES} files per rank"
            )
            activation_filepath_limit = min(
                activation_filepath_limit, self.NUM_DEBUG_FILES
            )

        local_activation_filepaths = activation_filepaths[
            dist.get_rank() : activation_filepath_limit : dist.get_world_size()
        ]

        # start threadpool to load the files
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = deque(
                executor.submit(th.load, filepath)
                for filepath in local_activation_filepaths
            )
            for future_idx in tqdm(
                range(
                    dist.get_rank(), activation_filepath_limit, dist.get_world_size()
                ),
                total=len(futures),
                desc="Loading batch sizes",
                leave=False,
                position=dist.get_rank(),
            ):
                future = futures.popleft()
                data = future.result()
                all_batch_sizes[future_idx] = data[ActivationKeys.MLP_OUTPUT].shape[0]
                del data
                del future

        # on nccl only gpu-gpu communication is supported
        all_batch_sizes = all_batch_sizes.to("cuda")
        dist.all_reduce(all_batch_sizes, op=dist.ReduceOp.SUM)
        all_batch_sizes = all_batch_sizes.to(self.device)

        # maybe not the bestest practice but this is good enough lol
        th.random.seed(seed + dist.get_rank())

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
            dist.get_rank() : activation_filepath_limit : dist.get_world_size()
        ]

        for shuffle_batch_idx, shuffle_batch in tqdm(
            enumerate(local_activation_file_batches),
            desc=f"Rank {dist.get_rank()}",
            total=len(local_activation_file_batches),
            leave=False,
            position=dist.get_rank() * 2,
        ):
            filepaths, batch_sizes = zip(*shuffle_batch, strict=True)

            file_data = asyncio.run(self.load_files_async(filepaths))

            batch_sizes = th.stack(batch_sizes, dim=0)
            batch_size_ranges = th.cumsum(batch_sizes, dim=0)
            total_size = batch_size_ranges[-1]

            batch_shuffled_indices = th.randperm(total_size.item())

            for batch_idx in tqdm(
                batch_shuffled_indices,
                desc=f"Shuffle batch {shuffle_batch_idx}",
                total=total_size,
                leave=False,
                position=dist.get_rank() * 2 + 1,
            ):
                file_idx, local_idx = Activations._batch_idx_to_file_and_local_idx(
                    batch_size_ranges, batch_idx
                )
                data = file_data[file_idx]

                for key, value in data.items():
                    match value:
                        case th.Tensor() | list():
                            current_batch[key].append(value[local_idx])
                        case _:
                            if key in current_batch:
                                assert current_batch[key] == value, (
                                    f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                )
                            else:
                                current_batch[key] = value

                num_batch_tokens += 1

                if num_batch_tokens >= tokens_per_file_in_reshuffled:
                    output_filepath = os.path.join(
                        output_dir, f"{dist.get_rank()}_{current_batch_idx}.pt-temp"
                    )
                    new_activation_filepaths.append(
                        self._collate_and_save_batch(current_batch, output_filepath)
                    )

                    current_batch = defaultdict(list)
                    total_tokens += num_batch_tokens
                    num_batch_tokens = 0
                    current_batch_idx += 1

        total_tokens += num_batch_tokens
        total_tokens = th.tensor(total_tokens, dtype=th.int32)
        dist.reduce(total_tokens, dst=0, op=dist.ReduceOp.SUM)

        self._total_tokens = total_tokens.item()

        remaining_batches = [None] * dist.get_world_size()

        if dist.get_rank() == 0:
            logger.info(f"Total tokens: {total_tokens.item()}")
            dist.gather_object(current_batch, remaining_batches, dst=0)

            non_empty_batches = [batch for batch in remaining_batches if batch]

            extra_activation_filepaths = []
            if len(non_empty_batches) > 0:
                concatenated_batch = non_empty_batches[0]
                for batch in non_empty_batches[1:]:
                    for key, value in batch.items():
                        if isinstance(value, list):
                            concatenated_batch[key].extend(value)
                        else:
                            concatenated_batch[key] = value

                total_extra_tokens = len(concatenated_batch["tokens"])
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
                        f"{dist.get_rank()}_{current_batch_idx + extra_batch_idx}.pt-temp",
                    )
                    extra_activation_filepaths.append(
                        self._collate_and_save_batch(extra_batch, output_filepath)
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

            renamed_activation_filepaths = [
                f"{i}.pt" for i in range(len(new_activation_filepaths))
            ]
        else:
            dist.gather_object(current_batch, dst=0)
            renamed_activation_filepaths = None

        renamed_activation_filepaths = broadcast_variable_length_list(
            lambda: renamed_activation_filepaths,
            src=0,
        )

        return renamed_activation_filepaths

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
        batch_size_ranges: th.Tensor, batch_idx: int
    ) -> tuple[int, int]:
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
) -> tuple[Activations, dict[str, int]]:
    """
    Load activations and initialize the distributed process group.

    Returns:
        activations: Activations object
        activation_dims: Dimensionality of the activations for each submodule as a dictionary
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

    logger.debug("Initializing distributed process group")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_distributed_logging()

    logger.debug(f"Initializing activations with seed {seed}")
    activations = Activations(
        experiment_name=activations_experiment_name,
        tokens_per_file_in_reshuffled=reshuffled_tokens_per_file,
        seed=seed,
        num_workers=num_workers,
        debug=debug,
    )

    # load a batch of activations to get the dimension
    data_iterable = activations(batch_size=1)
    activation = next(data_iterable)
    activation_dims = {
        submodule_name: th.prod(th.tensor(activation[submodule_name].shape[1:])).item()
        for submodule_name in submodule_names
    }

    # clean up the background worker and queue
    data_iterable.send("STOP!")

    return activations, activation_dims
