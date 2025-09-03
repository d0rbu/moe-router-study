from collections import defaultdict
from collections.abc import Iterator
from itertools import batched, count, pairwise
import os

import torch as th
import torch.multiprocessing as mp
from tqdm import tqdm

from core.slurm import SlurmEnv
from exp import ACTIVATION_DIRNAME, OUTPUT_DIR


class Activations:
    def __init__(
        self,
        experiment_name: str,
        slurm_env: SlurmEnv,
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
            seed: Seed for the random number generator
            max_cache_size: Maximum number of file data entries to cache
        """
        activation_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

        self.slurm_env = slurm_env
        self.device = device
        self.activation_filepaths = self.load_files(
            activation_dir=activation_dir,
            slurm_env=slurm_env,
            reshuffle=reshuffle,
            seed=seed,
            tokens_per_file=tokens_per_file_in_reshuffled,
            shuffle_batch_size=shuffle_batch_size,
        )
        self.max_cache_size = max_cache_size

    # worker to fetch data from disk
    def _get_file_data(self, cached_file_data: mp.Queue):
        for activation_filepath in self.activation_filepaths:
            file_data = th.load(activation_filepath)
            cached_file_data.put(file_data, block=True)
        cached_file_data.put(None)

    def __call__(self, batch_size: int = 4096, ctx_len: int = 128) -> Iterator[dict]:
        def data_generator(batch_size: int, ctx_len: int) -> Iterator[dict]:
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

            if ctx_len <= 0:
                ctx_len = (
                    1024 * 1024
                )  # probably not very future proof given the scary pace of progress

            current_batch = {}
            remaining_batch_size = batch_size

            for batch_idx in count():
                while current_data_size - current_local_idx < remaining_batch_size:
                    for key, value in current_data.items():
                        match value:
                            case th.Tensor():
                                if key in current_batch:
                                    current_batch[key] = th.cat(
                                        current_batch[key],
                                        value[
                                            current_local_idx : current_local_idx
                                            + batch_size,
                                            :,
                                            :ctx_len,
                                        ],
                                        dim=0,
                                    )
                                else:
                                    current_batch[key] = value[
                                        current_local_idx : current_local_idx
                                        + batch_size,
                                        :,
                                        :ctx_len,
                                    ]
                            case list():
                                truncated_sequences = [
                                    sequence[:ctx_len]
                                    for sequence in value[
                                        current_local_idx : current_local_idx
                                        + batch_size
                                    ]
                                ]
                                if key in current_batch:
                                    current_batch[key].extend(truncated_sequences)
                                else:
                                    current_batch[key] = truncated_sequences
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
                        return

                    current_local_idx = 0
                    current_data_size = len(current_data["tokens"])
                else:
                    for key, value in current_data.items():
                        match value:
                            case th.Tensor():
                                if key in current_batch:
                                    current_batch[key] = th.cat(
                                        current_batch[key],
                                        value[
                                            current_local_idx : current_local_idx
                                            + batch_size,
                                            :,
                                            :ctx_len,
                                        ],
                                        dim=0,
                                    )
                                else:
                                    current_batch[key] = value[
                                        current_local_idx : current_local_idx
                                        + batch_size,
                                        :,
                                        :ctx_len,
                                    ]
                            case list():
                                truncated_sequences = [
                                    sequence[:ctx_len]
                                    for sequence in value[
                                        current_local_idx : current_local_idx
                                        + batch_size
                                    ]
                                ]
                                if key in current_batch:
                                    current_batch[key].extend(truncated_sequences)
                                else:
                                    current_batch[key] = truncated_sequences
                            case _:
                                if key in current_batch:
                                    assert current_batch[key] == value, (
                                        f"Inconsistent value for {key}: {current_batch[key]} != {value}"
                                    )
                                else:
                                    current_batch[key] = value

                    if batch_idx % self.slurm_env.world_size == self.slurm_env.world_rank:
                        yield current_batch

                    current_batch = {}
                    current_local_idx += remaining_batch_size
                    remaining_batch_size = batch_size

        return data_generator(batch_size, ctx_len)

    def __iter__(self) -> Iterator[dict]:
        return self()

    @staticmethod
    def load_files(
        activation_dir: str,
        slurm_env: SlurmEnv,
        reshuffle: bool = False,
        seed: int = 0,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
    ) -> list[str]:
        if reshuffle:
            shuffle_dirname = f"reshuffled-seed={seed}-tokens_per_file={tokens_per_file_in_reshuffled}"
            activation_files_dir = os.path.join(activation_dir, shuffle_dirname)
            if slurm_env.world_rank == 0:
                os.makedirs(activation_files_dir, exist_ok=True)
        else:
            activation_files_dir = activation_dir

        activation_filepaths = Activations.get_activation_filepaths(
            activation_files_dir
        )

        if activation_filepaths:
            return activation_filepaths

        th.distributed.barrier()

        # if we are here then there are no activation files
        if reshuffle:
            num_new_activation_filepaths = [0]
            if slurm_env.world_rank == 0:
                new_activation_filepaths = Activations.reshuffle(
                    activation_dir=activation_dir,
                    output_dir=activation_files_dir,
                    tokens_per_file=tokens_per_file_in_reshuffled,
                    seed=seed,
                    shuffle_batch_size=shuffle_batch_size,
                )
                num_new_activation_filepaths[0] = len(new_activation_filepaths)

                th.distributed.broadcast_object_list(
                    num_new_activation_filepaths, src=0
                )
                th.distributed.broadcast_object_list(
                    new_activation_filepaths, src=0
                )
            else:
                th.distributed.broadcast_object_list(
                    num_new_activation_filepaths, src=0
                )
                new_activation_filepaths = [None] * num_new_activation_filepaths[0]
                th.distributed.broadcast_object_list(
                    new_activation_filepaths, src=0
                )

            return new_activation_filepaths

        raise FileNotFoundError(f"No activation files found in {activation_dir}")

    @staticmethod
    def get_activation_filepaths(activation_dir: str) -> list[str]:
        all_activation_filenames = [
            filename
            for filename in os.listdir(activation_dir)
            if filename.endswith(".pt")
        ]

        activation_indices = [
            int(filename.split(".")[0]) for filename in all_activation_filenames
        ]
        max_contiguous_activation_index = max(activation_indices)
        for prev_index, next_index in pairwise(activation_indices):
            if next_index - prev_index > 1:
                max_contiguous_activation_index = prev_index
                break

        contiguous_activation_filepaths = [
            os.path.join(activation_dir, f"{i}.pt")
            for i in range(max_contiguous_activation_index + 1)
        ]
        return contiguous_activation_filepaths

    @staticmethod
    def reshuffle(
        activation_dir: str,
        output_dir: str,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
        seed: int = 0,
    ) -> list[str]:
        activation_filepaths = Activations.get_activation_filepaths(activation_dir)
        batch_sizes = th.zeros(len(activation_filepaths), dtype=th.int32)

        for i, filepath in enumerate(activation_filepaths):
            with th.load(filepath) as data:
                batch_sizes[i] = len(data["tokens"])

        th.random.seed(seed)

        current_batch = defaultdict(list)
        current_batch_idx = 0
        num_batch_tokens = 0

        for shuffle_batch, batch_sizes in tqdm(
            batched(
                zip(activation_filepaths, batch_sizes, strict=False), shuffle_batch_size
            ),
            desc="Reshuffling",
            total=len(activation_filepaths) // shuffle_batch_size,
        ):
            file_data = [th.load(filepath) for filepath in shuffle_batch]

            batch_size_ranges = th.cumsum(batch_sizes, dim=0)
            total_size = batch_size_ranges[-1]

            batch_shuffled_indices = th.randperm(total_size)

            for batch_idx in tqdm(
                batch_shuffled_indices, total=total_size, leave=False
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

                num_batch_tokens += len(data["tokens"][local_idx])

                if num_batch_tokens >= tokens_per_file_in_reshuffled:
                    output_filepath = os.path.join(
                        output_dir, f"{current_batch_idx}.pt"
                    )
                    Activations._collate_and_save_batch(current_batch, output_filepath)

                    current_batch = defaultdict(list)
                    num_batch_tokens = 0
                    current_batch_idx += 1

        if current_batch:
            output_filepath = os.path.join(output_dir, f"{current_batch_idx}.pt")
            Activations._collate_and_save_batch(current_batch, output_filepath)

    @staticmethod
    def _collate_and_save_batch(batch: dict, output_filepath: str):
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                batch[key] = th.stack(value, dim=0)
        th.save(batch, output_filepath)

    @staticmethod
    def _batch_idx_to_file_and_local_idx(
        batch_size_ranges: th.Tensor, batch_idx: int
    ) -> tuple[int, int]:
        file_idx = th.searchsorted(batch_size_ranges, batch_idx, side="right")
        local_idx = batch_idx - batch_size_ranges[file_idx]
        return file_idx, local_idx
