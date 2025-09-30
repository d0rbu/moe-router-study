import os

from loguru import logger
import torch.distributed as dist


def init_distributed_logging(rank_filename_format: str | None = None) -> None:
    # assert that torch distributed is initialized
    if not hasattr(dist, "is_initialized"):
        raise RuntimeError("torch.distributed.is_initialized is not available")

    assert dist.is_initialized(), (
        "Torch distributed must be initialized first, call dist.init_process_group() before calling init_distributed_logging()"
    )

    if not hasattr(dist, "get_rank"):
        raise RuntimeError("torch.distributed.get_rank is not available")

    rank = dist.get_rank()

    if rank == 0:
        # log as normal
        return

    if rank_filename_format is None:
        # do not log anything
        logger.remove()
        return

    # make sure {rank} is in the filename format
    assert "{rank}" in rank_filename_format, (
        f"rank_filename_format must contain {{rank}}, currently it is {rank_filename_format}"
    )

    # log to a file
    rank_filename = rank_filename_format.format(rank=rank)
    os.makedirs(os.path.dirname(rank_filename), exist_ok=True)

    logger.trace(f"Rank {rank} logging to {rank_filename}")
    logger.remove()
    logger.add(rank_filename)
