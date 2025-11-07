# Extract kmeans_manhattan function from backup and insert it into the refactored file

with open("exp/kmeans.py.backup", "r") as f:
    backup_lines = f.readlines()

with open("exp/kmeans.py", "r") as f:
    current_lines = f.readlines()

# Find the kmeans_manhattan function in backup (lines 1078-1600, but 0-indexed so 1077-1599)
kmeans_func_lines = backup_lines[1077:1600]

# Remove 'async ' from the function definition
kmeans_func_lines[0] = kmeans_func_lines[0].replace("async def", "def")

# Remove all 'await ' keywords from the function
kmeans_func_lines = [line.replace("await ", "") for line in kmeans_func_lines]

# Replace asyncio.Queue with mp.Queue
kmeans_func_lines = [line.replace("asyncio.Queue", "mp.Queue") for line in kmeans_func_lines]

# Replace Barrier( with mp.Barrier(
import re
kmeans_func_lines = [re.sub(r'\bBarrier\(', 'mp.Barrier(', line) for line in kmeans_func_lines]

# Replace asyncio.create_task with mp.Process  
# Find the worker creation pattern and replace it
new_func_lines = []
i = 0
while i < len(kmeans_func_lines):
    line = kmeans_func_lines[i]
    
    # Check for asyncio.create_task pattern
    if "asyncio.create_task(" in line:
        # This should be the workers list comprehension
        # Collect the full expression
        expr_lines = [line]
        j = i + 1
        paren_count = line.count('(') - line.count(')')
        while j < len(kmeans_func_lines) and paren_count > 0:
            expr_lines.append(kmeans_func_lines[j])
            paren_count += kmeans_func_lines[j].count('(') - kmeans_func_lines[j].count(')')
            j += 1
        
        # Replace with mp.Process creation
        indent = line[:len(line) - len(line.lstrip())]
        new_func_lines.append(f"{indent}workers = []\n")
        new_func_lines.append(f"{indent}for gpu_idx in range(num_gpus):\n")
        new_func_lines.append(f"{indent}    p = mp.Process(\n")
        new_func_lines.append(f"{indent}        target=gpu_worker,\n")
        new_func_lines.append(f"{indent}        args=(\n")
        new_func_lines.append(f"{indent}            gpu_idx,\n")
        new_func_lines.append(f"{indent}            all_gpu_data,\n")
        new_func_lines.append(f"{indent}            top_k,\n")
        new_func_lines.append(f"{indent}            losses_over_time,\n")
        new_func_lines.append(f"{indent}            synchronization_barrier,\n")
        new_func_lines.append(f"{indent}            general_gpu_group,\n")
        new_func_lines.append(f"{indent}            gpu_process_groups[gpu_idx] if gpu_process_groups is not None else None,\n")
        new_func_lines.append(f"{indent}            save_dir,\n")
        new_func_lines.append(f"{indent}            validate_every,\n")
        new_func_lines.append(f"{indent}            centroid_minibatch_size,\n")
        new_func_lines.append(f"{indent}            assignment_minibatch_size,\n")
        new_func_lines.append(f"{indent}        ),\n")
        new_func_lines.append(f"{indent}        name=str(gpu_idx)\n")
        new_func_lines.append(f"{indent}    )\n")
        new_func_lines.append(f"{indent}    workers.append(p)\n")
        new_func_lines.append(f"{indent}\n")
        new_func_lines.append(f"{indent}# Start all workers\n")
        new_func_lines.append(f"{indent}for worker in workers:\n")
        new_func_lines.append(f"{indent}    worker.start()\n")
        i = j
        continue
    
    # Check for asyncio.gather(*workers)
    if "asyncio.gather(*workers)" in line:
        # Skip this line - we'll handle worker termination differently
        i += 1
        continue
    
    # Check for worker.add_done_callback
    if "add_done_callback" in line:
        # Skip this line
        i += 1
        continue
    
    new_func_lines.append(line)
    i += 1

# Now find where to insert the function in current file
# It should go before cluster_paths_main (line 1007 in current, 0-indexed 1006)
insert_pos = 1006

# Insert the function
new_content = current_lines[:insert_pos] + new_func_lines + ["\n\n"] + current_lines[insert_pos:]

with open("exp/kmeans.py", "w") as f:
    f.writelines(new_content)

print(f"Restored kmeans_manhattan function ({len(new_func_lines)} lines)")
