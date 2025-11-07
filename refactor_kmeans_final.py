#!/usr/bin/env python3
"""Final phase: Complete the async to multiprocessing refactoring."""

import re

with open("exp/kmeans.py", "r") as f:
    content = f.read()

# 1. Remove unused asyncio and Awaitable imports since we're not using them anymore
content = re.sub(r'import asyncio\n', '', content)
content = re.sub(r'from collections\.abc import Awaitable\n', '', content)

# 2. Remove the handle_exceptions import since asyncio tasks don't exist
content = re.sub(r'from core\.async_utils import handle_exceptions\n', '', content)

# 3. Fix the main function calls - replace asyncio.run with direct calls
# cluster_paths_async should just become cluster_paths_main
content = content.replace('def cluster_paths_async(', 'def cluster_paths_main(')
content = content.replace('asyncio.run(\n        cluster_paths_async(', 'cluster_paths_main(')

# Also fix the load_activations call
content = content.replace(
    'activations, activation_dims, gpu_process_group, gpu_process_groups = asyncio.run(\n        load_activations_and_init_dist(',
    'activations, activation_dims, gpu_process_group, gpu_process_groups = load_activations_and_init_dist('
)

# 4. Remove the safe__with_worker_check function entirely
safe_function_pattern = r'def safe_with_worker_check\[T\]\(.*?\n(?:.*?\n)*?.*?raise\n\n'
content = re.sub(safe_function_pattern, '', content, flags=re.DOTALL)

# 5. Replace worker startup - convert from asyncio tasks to processes
# The workers list creation needs to change
old_workers_pattern = r'workers = \[\s*mp\.Process\(target=gpu_worker, args=\((.*?)\), name=str\(gpu_idx\)\)\s*for gpu_idx in range\(num_gpus\)\s*\]'

new_workers = '''workers = []
    for gpu_idx in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(
                gpu_idx,
                all_gpu_data,
                top_k,
                losses_over_time,
                synchronization_barrier,
                general_gpu_group,
                gpu_process_groups[gpu_idx] if gpu_process_groups is not None else None,
                save_dir,
                validate_every,
                centroid_minibatch_size,
                assignment_minibatch_size,
                device_type,
            ),
            name=str(gpu_idx),
        )
        workers.append(p)'''

# Find and replace the workers creation
workers_match = re.search(r'    workers = \[.*?for gpu_idx in range\(num_gpus\)\s*\]', content, re.DOTALL)
if workers_match:
    content = content.replace(workers_match.group(0), new_workers)

# 6. Start the workers
old_pattern = r'    logger\.trace\(f"Created \{len\(workers\)\} workers"\)'
new_pattern = '''    logger.trace(f"Created {len(workers)} workers")
    
    # Start all worker processes
    for worker in workers:
        worker.start()
    
    logger.trace(f"Started {len(workers)} worker processes")'''
content = content.replace('    logger.trace(f"Created {len(workers)} workers")', new_pattern)

# 7. At the end of the iteration loop, we need to terminate workers properly
# Find the section where workers need to be stopped
stop_signal_pattern = r'(\s+)# send stop signal to workers'
if re.search(stop_signal_pattern, content):
    # Add worker joining after stop signals
    content = re.sub(
        r'(\s+)# send stop signal to workers\n(\s+)for gpu_data in all_gpu_data:\n(\s+)gpu_data\.queue\.put\(None\)',
        r'\1# send stop signal to workers\n\2for gpu_data in all_gpu_data:\n\3gpu_data.queue.put(None)\n\n\2# Wait for all workers to finish\n\2for worker in workers:\n\3worker.join(timeout=10.0)\n\3if worker.is_alive():\n\4logger.warning(f"Worker {worker.name} did not terminate gracefully, killing it")\n\4worker.terminate()\n\4worker.join()',
        content
    )

# 8. Fix the  @dataclass for GPUData to use correct queue type
content = content.replace(
    'queue: asyncio.Queue',
    'queue: mp.Queue'
)

# Write the result
with open("exp/kmeans.py", "w") as f:
    f.write(content)

print("Final refactoring complete!")
print("\nRemaining manual fixes needed:")
print("- Check that all worker process args are correctly passed")
print("- Verify Queue operations work without await")
print("- Test that barrier synchronization works across processes")

