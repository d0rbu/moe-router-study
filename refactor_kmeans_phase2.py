#!/usr/bin/env python3
"""Phase 2: Handle remaining async issues in kmeans.py."""

import re

# Read the current file
with open("exp/kmeans.py", "r") as f:
    lines = f.readlines()

# Process line by line for complex replacements
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Replace asyncio.wait_for for Queue.get()
    if "queue_item = asyncio.wait_for(gpu_data.queue.get(), timeout=60.0)" in line:
        # Replace with blocking get with timeout
        output_lines.append(line.replace(
            "queue_item = asyncio.wait_for(gpu_data.queue.get(), timeout=60.0)",
            "queue_item = gpu_data.queue.get(timeout=60.0)"
        ))
        i += 1
        continue
    
    # Replace asyncio.gather for kmeans_step
    if "updates = asyncio.gather(" in line and i < len(lines) - 6:
        # Found the asyncio.gather block for kmeans_step
        # Replace with list comprehension (sequential execution)
        output_lines.append("        updates = [\n")
        output_lines.append("            kmeans_step(\n")
        i += 2  # Skip the asyncio.gather and next line
        # Copy the kmeans_step arguments
        while i < len(lines) and "for centroids in" not in lines[i]:
            output_lines.append(lines[i])
            i += 1
        # Now handle the for loop
        if i < len(lines):
            # Change from generator to list comprehension
            for_line = lines[i].replace("*[", "").strip()
            output_lines.append(f"            {for_line}\n")
            i += 1
            # Skip the closing ]
            while i < len(lines) and "]" in lines[i] and ")" in lines[i]:
                output_lines.append("        ]\n")
                i += 1
                break
        continue
    
    # Replace asyncio.gather for dist.wait() calls
    if "asyncio.gather(" in line and "asyncio.to_thread" in lines[i+1] if i+1 < len(lines) else False:
        # This is the distributed waiting - replace with sequential calls
        indent = len(line) - len(line.lstrip())
        output_lines.append(" " * indent + "centroids_future.wait()\n")
        output_lines.append(" " * indent + "weights_future.wait()\n")
        # Skip until we find the closing parenthesis
        while i < len(lines) and not ("))" in lines[i] or "),)" in lines[i]):
            i += 1
        i += 1  # Skip the closing line
        continue
    
    # Skip the safe_await_with_worker_check function entirely - we'll rewrite it
    if "def safe_with_worker_check" in line:
        # Skip until the end of this function
        while i < len(lines):
            i += 1
            if i < len(lines) and lines[i].strip() and not lines[i].startswith(" ") and not lines[i].startswith("\t"):
                break
        continue
    
    # Replace calls to safe__with_worker_check
    if "safe_with_worker_check(" in line:
        # Extract just the inner function call
        # Pattern: safe_with_worker_check(actual_call(...), workers=..., ...)
        match = re.search(r'safe_with_worker_check\(\s*([^,]+(?:\([^)]*\))?)', line)
        if match:
            inner_call = match.group(1).strip()
            indent = len(line) - len(line.lstrip())
            output_lines.append(" " * indent + inner_call + "\n")
            # Skip continuation lines if any
            paren_count = line.count('(') - line.count(')')
            while paren_count > 0 and i + 1 < len(lines):
                i += 1
                paren_count += lines[i].count('(') - lines[i].count(')')
            i += 1
            continue
    
    # Keep the line as is
    output_lines.append(line)
    i += 1

# Write back
with open("exp/kmeans.py", "w") as f:
    f.writelines(output_lines)

print("Phase 2 refactoring complete!")

