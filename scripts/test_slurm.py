print("Starting test_slurm.py")

from datetime import timedelta  # noqa: E402
import os  # noqa: E402

print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"))
print("MASTER_PORT:", os.environ.get("MASTER_PORT"))
print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))
print("SLURM_JOB_NODELIST:", os.environ.get("SLURM_JOB_NODELIST"))
print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
print("SLURM_NTASKS_PER_NODE:", os.environ.get("SLURM_NTASKS_PER_NODE"))
print("SLURM_PROCID:", os.environ.get("SLURM_PROCID"))
print("SLURM_LOCALID:", os.environ.get("SLURM_LOCALID"))
print("SLURM_NODEID:", os.environ.get("SLURM_NODEID"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
print("RANK:", os.environ.get("RANK"))
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))

import torch.distributed as dist  # noqa: E402

dist.init_process_group(
    backend="nccl", timeout=timedelta(seconds=30)
)

print(f"[Rank {dist.get_rank()}] Hello from {os.uname().nodename}")
dist.barrier()
if dist.get_rank() == 0:
    print("PMI rendezvous successful.")

dist.destroy_process_group()
