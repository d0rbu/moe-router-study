import re

# Read the backup file
with open("exp/kmeans.py.backup", "r") as f:
    content = f.read()

# Phase 1: Remove asyncio imports and add multiprocessing
content = content.replace("import asyncio\n", "")
content = content.replace("from asyncio import Barrier\n", "")
content = content.replace("from collections.abc import Awaitable\n", "")
content = content.replace("from core.async_utils import handle_exceptions\n", "")

# Add multiprocessing import after torch imports
import_section_end = content.find("from loguru import logger")
if "import torch.multiprocessing as mp" not in content:
    content = content[:import_section_end] + "import torch.multiprocessing as mp\n" + content[import_section_end:]

# Phase 2: Remove async/await keywords
# Remove 'async ' before 'def'
content = re.sub(r'\basync\s+def\b', 'def', content)

# Remove 'await ' before function calls
content = re.sub(r'\bawait\s+', '', content)

# Phase 3: Replace asyncio constructs with multiprocessing equivalents
# Replace asyncio.Queue with mp.Queue
content = content.replace("asyncio.Queue", "mp.Queue")

# Replace Barrier with mp.Barrier
content = re.sub(r'\bBarrier\(', 'mp.Barrier(', content)

# Replace asyncio.Task type annotations with mp.Process
content = content.replace("asyncio.Task", "mp.Process")

# Phase 4: Remove safe_await_with_worker_check function
# Find and remove the entire function definition
safe_await_pattern = r'def safe_await_with_worker_check\[T\]\(.*?\n    raise\n\n\n'
content = re.sub(safe_await_pattern, '', content, flags=re.DOTALL)

# Phase 5: Replace calls to safe_await_with_worker_check
# This is tricky - we need to extract just the function call and remove the wrapper
# Pattern: safe_await_with_worker_check(function_call(...), workers=..., timeout=..., operation_name=...)
# Should become just: function_call(...)

def replace_safe_await(match):
    # Extract the actual function call (first argument)
    full_match = match.group(0)
    # Find the first opening paren after safe_await_with_worker_check
    start_idx = full_match.find('(') + 1
    # Now we need to find the matching closing paren for the actual function call
    # This is complex because the function call itself has parens
    
    # Simple heuristic: find '),\n' which typically separates the function call from keywords
    end_marker = '),\n'
    end_idx = full_match.find(end_marker)
    
    if end_idx != -1:
        func_call = full_match[start_idx:end_idx+1]
        return func_call.strip()
    
    return full_match  # Fallback

# Apply the replacement
content = re.sub(
    r'safe_await_with_worker_check\((.*?)\),\s*workers=.*?operation_name=.*?\)',
    replace_safe_await,
    content,
    flags=re.DOTALL
)

# Phase 6: Fix asyncio.gather patterns for kmeans_step
# Replace: asyncio.gather(*[kmeans_step(...) for ...])
# With: [kmeans_step(...) for ...]
gather_pattern = r'asyncio\.gather\(\*(\[.*?\])\)'
def replace_gather(match):
    return match.group(1)  # Just return the list comprehension

content = re.sub(gather_pattern, replace_gather, content, flags=re.DOTALL)

# Phase 7: Fix asyncio.wait_for with queue.get
# Replace: asyncio.wait_for(queue.get(), timeout=X)
# With: queue.get(timeout=X)
content = re.sub(
    r'asyncio\.wait_for\((\w+\.queue\.get)\(\),\s*timeout=([^)]+)\)',
    r'\1(timeout=\2)',
    content
)

# Phase 8: Fix asyncio.to_thread(future.wait) - just remove the asyncio.to_thread wrapper
content = re.sub(r'asyncio\.to_thread\(([^)]+\.wait)\)', r'\1()', content)

# Phase 9: Replace asyncio.create_task with mp.Process creation
# This is complex - we need to change the structure significantly
# From: asyncio.create_task(gpu_worker(...), name=str(gpu_idx))
# To: mp.Process(target=gpu_worker, args=(...), name=str(gpu_idx))

# First, find the worker creation section
worker_creation_pattern = r'workers = \[\s*asyncio\.create_task\(\s*gpu_worker\((.*?)\),\s*name=str\(gpu_idx\),\s*\)\s*for gpu_idx in range\(num_gpus\)\s*\]'

def replace_worker_creation(match):
    args = match.group(1)
    return f'''workers = []
    for gpu_idx in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=({args}),
            name=str(gpu_idx)
        )
        workers.append(p)
    
    # Start all workers
    for worker in workers:
        worker.start()'''

content = re.sub(worker_creation_pattern, replace_worker_creation, content, flags=re.DOTALL)

# Phase 10: Replace worker exception handling
# From: worker.add_done_callback(handle_exceptions)
# To: (remove - not needed for multiprocessing)
content = re.sub(r'\s*for worker in workers:\s*worker\.add_done_callback\(handle_exceptions\)\s*\n', '', content)

# Phase 11: Replace asyncio.gather(*workers) with proper worker joining
# From: asyncio.gather(*workers)
# To: for worker in workers: worker.join()
content = content.replace('asyncio.gather(*workers)', '')

# Phase 12: Rename cluster_paths_async to cluster_paths_main
content = content.replace('cluster_paths_async', 'cluster_paths_main')

# Phase 13: Remove asyncio.run() wrappers
content = re.sub(r'asyncio\.run\((.*?)\)', r'\1', content)

# Phase 14: Fix check_worker_health function
# Replace task.done() and task.exception() with process.is_alive() and exitcode
old_check = '''def check_worker_health(workers: dict[str, mp.Process] | list[mp.Process]) -> None:
    """Check if any workers have failed and log their status."""
    worker_dict = workers if isinstance(workers, dict) else {str(i): w for i, w in enumerate(workers)}
    
    for worker_name, worker in worker_dict.items():
        if worker.done():
            try:
                exception = worker.exception()
                if exception is not None:
                    logger.error(
                        f"Worker {worker_name} failed with exception: {exception}"
                    )
                else:
                    logger.error(f"Worker {worker_name} completed unexpectedly")
            except Exception as e:
                logger.error(f"Error checking worker {worker_name}: {e}")'''

new_check = '''def check_worker_health(workers: dict[str, mp.Process] | list[mp.Process]) -> None:
    """Check if any workers have failed and log their status."""
    worker_dict = workers if isinstance(workers, dict) else {str(i): w for i, w in enumerate(workers)}
    
    for worker_name, worker in worker_dict.items():
        if not worker.is_alive():
            exitcode = worker.exitcode
            if exitcode is not None and exitcode != 0:
                logger.error(
                    f"Worker {worker_name} failed with exit code: {exitcode}"
                )
            else:
                logger.error(f"Worker {worker_name} completed unexpectedly")'''

if old_check in content:
    content = content.replace(old_check, new_check)

# Write the refactored content
with open("exp/kmeans.py", "w") as f:
    f.write(content)

print("Refactoring complete!")
