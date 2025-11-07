#!/usr/bin/env python3
"""Refactor kmeans.py from asyncio to multiprocessing."""

import re

# Read the original file
with open("exp/kmeans.py.backup", "r") as f:
    content = f.read()

# 1. Update imports - replace Barrier import
content = content.replace(
    "from asyncio import Barrier",
    "import torch.multiprocessing as mp"
)

# 2. Remove async keyword from function definitions
# Match async def function_name(...):
content = re.sub(r'\basync def\b', 'def', content)

# 3. Remove await keywords
content = re.sub(r'\bawait\s+', '', content)

# 4. Replace asyncio.Queue with mp.Queue in GPUData
content = content.replace(
    "queue: asyncio.Queue",
    "queue: mp.Queue"
)

# 5. Replace asyncio.Queue creation in initialization
content = content.replace(
    "asyncio.Queue(maxsize=GPU_QUEUE_MAXSIZE)",
    "mp.Queue(maxsize=GPU_QUEUE_MAXSIZE)"
)

# 6. Replace asyncio.Barrier with mp.Barrier
content = content.replace("Barrier(", "mp.Barrier(")
content = content.replace("barrier: Barrier", "barrier: mp.Barrier")

# 7. Replace asyncio.create_task with mp.Process
# This is more complex - need to handle the specific pattern
task_pattern = r'asyncio\.create_task\(\s*gpu_worker\((.*?)\),\s*name=str\(gpu_idx\),\s*\)'
process_replacement = r'mp.Process(target=gpu_worker, args=(\1), name=str(gpu_idx))'
content = re.sub(task_pattern, process_replacement, content, flags=re.DOTALL)

# 8. Replace worker.add_done_callback(handle_exceptions) with pass
# In multiprocessing, we can't use done_callback the same way
content = re.sub(r'\s+for worker in workers:\s+worker\.add_done_callback\(handle_exceptions\)', '', content)

# 9. Replace safe_await_with_worker_check calls
# These need to become simple synchronous calls
#content = re.sub(r'safe_await_with_worker_check\((.*?),\s*workers=workers_dict.*?\)', r'\1', content, flags=re.DOTALL)

# 10. Replace asyncio.gather with sequential execution
# asyncio.gather(*[kmeans_step(...)]) -> list of calls
gather_pattern = r'asyncio\.gather\(\s*\*\[\s*(kmeans_step\(.*?\))\s+for\s+(.*?)\s+in\s+(.*?)\]\s*\)'
# This one is tricky, let's handle it differently

# 11. Replace check_worker_health to work with processes
content = content.replace(
    "def check_worker_health(workers: dict[str, asyncio.Task], *, context: str = \"\") -> None:",
    "def check_worker_health(workers: dict[str, mp.Process], *, context: str = \"\") -> None:"
)

# 12. Update the check_worker_health logic for processes
old_health_check = """def check_worker_health(workers: dict[str, mp.Process], *, context: str = "") -> None:
    \"\"\"Check if any workers have failed and raise appropriate exceptions.\"\"\"
    for worker_name, worker in workers.items():
        if worker.done():
            exception = worker.exception()
            context_str = f" [{context}]" if context else ""
            if exception:
                logger.error(f"{worker_name} worker failed{context_str}: {exception}")
                raise RuntimeError(f"{worker_name} worker failed") from exception
            else:
                logger.error(
                    f"{worker_name} worker completed unexpectedly{context_str}"
                )
                raise RuntimeError(f"{worker_name} worker completed unexpectedly")"""

new_health_check = """def check_worker_health(workers: dict[str, mp.Process], *, context: str = "") -> None:
    \"\"\"Check if any workers have failed and raise appropriate exceptions.\"\"\"
    for worker_name, worker in workers.items():
        if not worker.is_alive() and worker.exitcode is not None and worker.exitcode != 0:
            context_str = f" [{context}]" if context else ""
            logger.error(f"{worker_name} worker failed{context_str} with exit code {worker.exitcode}")
            raise RuntimeError(f"{worker_name} worker failed with exit code {worker.exitcode}")"""

content = content.replace(old_health_check, new_health_check)

# Write the refactored file
with open("exp/kmeans.py", "w") as f:
    f.write(content)

print("Refactoring complete!")
print("\nNote: Manual adjustments still needed for:")
print("- asyncio.gather replacement in gpu_worker")
print("- safe_await_with_worker_check calls")
print("- Worker startup/shutdown logic")
print("- Queue get() timeout handling")

