from collections import defaultdict
from collections.abc import Generator
import gc
from itertools import batched, count, pairwise
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
        tokens_per_file_in_reshuffled: int = 10_000,
        shuffle_batch_size: int = 10,
        seed: int = 0,
        max_cache_size: int = 2,
    ):
        """
        Args:
            experiment_name: Name of the experiment
            device: Device to return the activations on
            tokens_per_file_in_reshuffled: Number of tokens per file, only used if reshuffling
            shuffle_batch_size: How many batches to shuffle at a time
            seed: Seed for the random number generator
            max_cache_size: Maximum number of file data entries to cache when iterating
        """
        activation_dir = os.path.join(OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME)

        self.device = device
        self.activation_filepaths = self.load_files(
            activation_dir=activation_dir,
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
                                    current_batch[key],
                                    value[
                                        current_local_idx : current_local_idx
                                        + batch_size
                                    ].to(self.device),
                                    dim=0,
                                )
                            else:
                                current_batch[key] = value[
                                    current_local_idx : current_local_idx
                                    + batch_size
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
                                    current_batch[key],
                                    value[current_local_idx : current_local_idx + batch_size].to(self.device),
                                    dim=0,
                                )
                            else:
                                current_batch[key] = value[current_local_idx : current_local_idx + batch_size].to(self.device)
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

    @staticmethod
    def load_files(
        activation_dir: str,
        seed: int = 0,
        tokens_per_file_in_reshuffled: int = 100_000,
        shuffle_batch_size: int = 100,
    ) -> list[str]:
        shuffle_dirname = (
            f"reshuffled-seed={seed}-tokens_per_file={tokens_per_file_in_reshuffled}"
        )
        activation_files_dir = os.path.join(activation_dir, shuffle_dirname)
        os.makedirs(activation_files_dir, exist_ok=True)

        activation_filepaths = Activations.get_activation_filepaths(
            activation_files_dir
        )

        if activation_filepaths:
            return activation_filepaths

        # if we are here then there are no activation files
        new_activation_filepaths = Activations.reshuffle(
            activation_dir=activation_dir,
            output_dir=activation_files_dir,
            tokens_per_file_in_reshuffled=tokens_per_file_in_reshuffled,
            seed=seed,
            shuffle_batch_size=shuffle_batch_size,
        )

        return new_activation_filepaths

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
        new_activation_filepaths = []
        activation_filepaths = Activations.get_activation_filepaths(activation_dir)
        all_batch_sizes = th.empty(len(activation_filepaths), dtype=th.int32)

        for i, filepath in enumerate(activation_filepaths):
            with th.load(filepath) as data:
                all_batch_sizes[i] = data["mlp_output"].shape[0]

        th.random.seed(seed)

        current_batch = defaultdict(list)
        current_batch_idx = 0
        total_tokens = 0
        num_batch_tokens = 0

        for shuffle_batch, batch_sizes in tqdm(
            batched(
                zip(activation_filepaths, all_batch_sizes, strict=False),
                shuffle_batch_size,
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

                num_batch_tokens += 1

                if num_batch_tokens >= tokens_per_file_in_reshuffled:
                    output_filepath = os.path.join(
                        output_dir, f"{current_batch_idx}.pt-temp"
                    )
                    new_activation_filepaths.append(
                        Activations._collate_and_save_batch(
                            current_batch, output_filepath
                        )
                    )

                    current_batch = defaultdict(list)
                    total_tokens += num_batch_tokens
                    num_batch_tokens = 0
                    current_batch_idx += 1

        if current_batch:
            output_filepath = os.path.join(output_dir, f"{current_batch_idx}.pt-temp")
            new_activation_filepaths.append(
                Activations._collate_and_save_batch(current_batch, output_filepath)
            )
            total_tokens += num_batch_tokens

        reshuffled_indices = th.randperm(
            len(new_activation_filepaths), generator=th.Generator().manual_seed(seed)
        )

        for new_idx, filepath in tqdm(
            zip(reshuffled_indices, new_activation_filepaths, strict=True),
            desc="Reshuffling output files",
            leave=False,
            total=len(new_activation_filepaths),
        ):
            os.rename(filepath, os.path.join(output_dir, f"{new_idx}.pt"))

        return new_activation_filepaths

    @staticmethod
    def _collate_and_save_batch(batch: dict, output_filepath: str) -> str:
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                batch[key] = th.stack(value, dim=0)
        th.save(batch, output_filepath)

        return output_filepath

    @staticmethod
    def _batch_idx_to_file_and_local_idx(
        batch_size_ranges: th.Tensor, batch_idx: int
    ) -> tuple[int, int]:
        file_idx = th.searchsorted(batch_size_ranges, batch_idx, side="right")
        local_idx = batch_idx - batch_size_ranges[file_idx]
        return file_idx, local_idx
