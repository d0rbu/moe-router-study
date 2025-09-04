from datetime import timedelta
import os

import torch.distributed as dist

print("RANK:", os.environ.get("RANK"))
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))

dist.init_process_group(
    backend="gloo", init_method="env://", timeout=timedelta(seconds=30)
)

print(f"[Rank {dist.get_rank()}] Hello from {os.uname().nodename}")
dist.barrier()
if dist.get_rank() == 0:
    print("PMI rendezvous successful.")

dist.destroy_process_group()
