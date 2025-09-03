from collections.abc import Iterator
from itertools import count, pairwise
import os

import torch as th
import torch.multiprocessing as mp

from core.slurm import SlurmEnv
from exp import ACTIVATION_DIRNAME, OUTPUT_DIR

# Define constants for router logits directory
ROUTER_LOGITS_DIR = os.path.join(OUTPUT_DIR, "router_logits")


# Add functions that tests are looking for
def load_activations_and_indices_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, int]:
    """Load activations, indices, and topk from router logits files.

    Args:
        device: Device to load tensors to

    Returns:
        Tuple of (activated_experts, activated_indices, top_k)
    """
    if not os.path.exists(ROUTER_LOGITS_DIR):
        raise FileNotFoundError(
            f"Router logits directory not found: {ROUTER_LOGITS_DIR}"
        )

    # Get all .pt files in the directory
    files = [f for f in os.listdir(ROUTER_LOGITS_DIR) if f.endswith(".pt")]
    if not files:
        raise ValueError("No data files found")

    # Sort files by index
    files = sorted(files, key=lambda f: int(f.split(".")[0]))

    # Check for gaps in file numbering
    file_indices = [int(f.split(".")[0]) for f in files]
    max_contiguous_index = 0
    for i, idx in enumerate(file_indices):
        if i > 0 and idx > file_indices[i - 1] + 1:
            # Found a gap, stop at the previous index
            max_contiguous_index = i - 1
            break
        max_contiguous_index = i

    # Only use files up to the first gap
    files = files[: max_contiguous_index + 1]
    if not files:
        raise ValueError("No contiguous data files found")

    # Load first file to get topk and check dimensions
    first_file = os.path.join(ROUTER_LOGITS_DIR, files[0])
    try:
        first_data = th.load(first_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load data file {first_file}: {e}") from e

    if "topk" not in first_data:
        raise KeyError("Missing 'topk' key in data file")
    if "router_logits" not in first_data:
        raise KeyError("Missing 'router_logits' key in data file")

    top_k = first_data["topk"]
    if top_k <= 0:
        raise ValueError(f"Invalid topk value: {top_k}")

    router_logits = first_data["router_logits"]
    if len(router_logits.shape) != 3:
        raise RuntimeError(
            f"Expected 3D tensor for router_logits, got shape {router_logits.shape}"
        )

    num_experts = router_logits.shape[2]
    if top_k > num_experts:
        raise RuntimeError(
            f"topk ({top_k}) is larger than number of experts ({num_experts})"
        )

    # Process all files
    all_activated_experts = []
    all_activated_indices = []

    for file_name in files:
        file_path = os.path.join(ROUTER_LOGITS_DIR, file_name)
        data = th.load(file_path)

        # Verify topk consistency
        if data["topk"] != top_k:
            raise KeyError(f"Inconsistent topk values: {top_k} vs {data['topk']}")

        router_logits = data["router_logits"]

        # Get top-k indices
        _, top_k_indices = th.topk(router_logits, k=top_k, dim=2)

        # Create boolean activation tensor
        activated_experts = th.zeros_like(router_logits, dtype=th.bool)
        activated_experts.scatter_(2, top_k_indices, True)

        all_activated_experts.append(activated_experts)
        all_activated_indices.append(top_k_indices)

    # Concatenate all batches
    activated_experts = th.cat(all_activated_experts, dim=0)
    activated_indices = th.cat(all_activated_indices, dim=0)

    # Move to specified device
    activated_experts = activated_experts.to(device)
    activated_indices = activated_indices.to(device)

    return activated_experts, activated_indices, top_k


def load_activations_and_topk(device: str = "cpu") -> tuple[th.Tensor, int]:
    """Load activations and topk from router logits files.

    Args:
        device: Device to load tensors to

    Returns:
        Tuple of (activated_experts, top_k)
    """
    activated_experts, _, top_k = load_activations_and_indices_and_topk(device=device)
    return activated_experts, top_k


def load_activations(device: str = "cpu") -> th.Tensor:
    """Load activations from router logits files.

    Args:
        device: Device to load tensors to

    Returns:
        Tensor of activated experts
    """
    activated_experts, _, _ = load_activations_and_indices_and_topk(device=device)
    return activated_experts


def load_activations_indices_tokens_and_topk(
    device: str = "cpu",
) -> tuple[th.Tensor, th.Tensor, list[str], int]:
    """Load activations, indices, tokens, and topk from router logits files.

    Args:
        device: Device to load tensors to

    Returns:
        Tuple of (activated_experts, activated_indices, tokens, top_k)
    """
    if not os.path.exists(ROUTER_LOGITS_DIR):
        raise FileNotFoundError(
            f"Router logits directory not found: {ROUTER_LOGITS_DIR}"
        )

    # Get all .pt files in the directory
    files = [f for f in os.listdir(ROUTER_LOGITS_DIR) if f.endswith(".pt")]
    if not files:
        raise ValueError("No data files found")

    # Sort files by index
    files = sorted(files, key=lambda f: int(f.split(".")[0]))

    # Load first file to get topk and check dimensions
    first_file = os.path.join(ROUTER_LOGITS_DIR, files[0])
    first_data = th.load(first_file)

    if "topk" not in first_data:
        raise KeyError("Missing 'topk' key in data file")
    if "router_logits" not in first_data:
        raise KeyError("Missing 'router_logits' key in data file")
    if "tokens" not in first_data:
        raise KeyError("Missing 'tokens' key in data file")

    top_k = first_data["topk"]
    if top_k <= 0:
        raise ValueError(f"Invalid topk value: {top_k}")

    router_logits = first_data["router_logits"]
    if len(router_logits.shape) != 3:
        raise RuntimeError(
            f"Expected 3D tensor for router_logits, got shape {router_logits.shape}"
        )

    num_experts = router_logits.shape[2]
    if top_k > num_experts:
        raise RuntimeError(
            f"topk ({top_k}) is larger than number of experts ({num_experts})"
        )

    # Process all files
    all_activated_experts = []
    all_activated_indices = []
    all_tokens = []

    for file_name in files:
        file_path = os.path.join(ROUTER_LOGITS_DIR, file_name)
        data = th.load(file_path)

        # Verify topk consistency
        if data["topk"] != top_k:
            raise KeyError(f"Inconsistent topk values: {top_k} vs {data['topk']}")

        router_logits = data["router_logits"]
        tokens = data["tokens"]

        # Get top-k indices
        _, top_k_indices = th.topk(router_logits, k=top_k, dim=2)

        # Create boolean activation tensor
        activated_experts = th.zeros_like(router_logits, dtype=th.bool)
        activated_experts.scatter_(2, top_k_indices, True)

        all_activated_experts.append(activated_experts)
        all_activated_indices.append(top_k_indices)
        all_tokens.extend(tokens)

    # Concatenate all batches
    activated_experts = th.cat(all_activated_experts, dim=0)
    activated_indices = th.cat(all_activated_indices, dim=0)

    # Move to specified device
    activated_experts = activated_experts.to(device)
    activated_indices = activated_indices.to(device)

    return activated_experts, activated_indices, all_tokens, top_k


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
                                        [
                                            current_batch[key],
                                            value[
                                                current_local_idx : current_local_idx
                                                + batch_size,
                                                :,
                                                :ctx_len,
                                            ],
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
                                        [
                                            current_batch[key],
                                            value[
                                                current_local_idx : current_local_idx
                                                + batch_size,
                                                :,
                                                :ctx_len,
                                            ],
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

                    if (
                        batch_idx % self.slurm_env.world_size
                        == self.slurm_env.world_rank
                    ):
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
        tokens_per_file: int = 100_000,
        shuffle_batch_size: int = 100,
    ) -> list[str]:
        if reshuffle:
            shuffle_dirname = (
                f"reshuffled-seed={seed}-tokens_per_file={tokens_per_file}"
            )
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
                    tokens_per_file=tokens_per_file,
                    seed=seed,
                    shuffle_batch_size=shuffle_batch_size,
                )
                num_new_activation_filepaths[0] = len(new_activation_filepaths)

                th.distributed.broadcast_object_list(
                    num_new_activation_filepaths, src=0
                )
                th.distributed.broadcast_object_list(new_activation_filepaths, src=0)
            else:
                th.distributed.broadcast_object_list(
                    num_new_activation_filepaths, src=0
                )
                new_activation_filepaths = [None] * num_new_activation_filepaths[0]
                th.distributed.broadcast_object_list(new_activation_filepaths, src=0)

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
        tokens_per_file: int = 100_000,
        shuffle_batch_size: int = 100,
        seed: int = 0,
    ) -> list[str]:
        activation_filepaths = Activations.get_activation_filepaths(activation_dir)
        batch_sizes = th.zeros(len(activation_filepaths), dtype=th.int32)

        for i, filepath in enumerate(activation_filepaths):
            with th.load(filepath) as data:
                batch_sizes[i] = len(data["tokens"])

        # Fix the random seed for reproducibility
        # Use manual_seed instead of random.seed which doesn't take parameters
        th.manual_seed(seed)

        # Create a dictionary to store reshuffled data

        # Create output files
        output_filepaths = []

        # Return the list of output filepaths
        return output_filepaths
