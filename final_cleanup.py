import re

with open("exp/kmeans.py", "r") as f:
    lines = f.readlines()

# Remove safe_await_with_worker_check function (lines 47-88)
new_lines = []
skip_until = -1
for i, line in enumerate(lines):
    if i < skip_until:
        continue
    
    # Find and skip the safe_await_with_worker_check function
    if "def safe_await_with_worker_check" in line:
        # Skip until we find the next function or class definition
        j = i + 1
        while j < len(lines):
            if lines[j].startswith("def ") or lines[j].startswith("class ") or lines[j].startswith("@"):
                skip_until = j
                break
            j += 1
        continue
    
    # Replace Barrier type annotation with mp.Barrier
    if "barrier: Barrier," in line:
        line = line.replace("barrier: Barrier,", "barrier: mp.Barrier,")
    
    # Replace asyncio.gather with list comprehension evaluation
    if "asyncio.gather(" in line:
        # Check if this is the centroids_future/weights_future.wait() pattern
        if i + 2 < len(lines) and "centroids_future.wait()" in lines[i+1] and "weights_future.wait()" in lines[i+2]:
            # Replace with direct wait calls
            new_lines.append("        centroids_future.wait()\n")
            new_lines.append("        weights_future.wait()\n")
            # Skip the asyncio.gather block
            j = i + 1
            while j < len(lines) and ")" not in lines[j]:
                j += 1
            skip_until = j + 1
            continue
        # Otherwise it's the kmeans_step gather - convert to list evaluation
        elif "*[" in line or (i + 1 < len(lines) and "*[" in lines[i+1]):
            # Find the full gather expression
            gather_lines = [line]
            j = i + 1
            paren_count = line.count('(') - line.count(')')
            while j < len(lines) and paren_count > 0:
                gather_lines.append(lines[j])
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1
            
            # Extract just the list comprehension
            full_gather = ''.join(gather_lines)
            # Find the list comprehension [...]
            start = full_gather.find('[')
            end = full_gather.rfind(']')
            if start != -1 and end != -1:
                list_comp = full_gather[start:end+1]
                # Get the indentation
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}updates = {list_comp}\n")
            skip_until = j
            continue
    
    # Remove asyncio.run() wrapper
    if "asyncio.run(" in line:
        # Find the matching closing paren
        content_lines = [line]
        j = i + 1
        paren_count = line.count('(') - line.count(')')
        while j < len(lines) and paren_count > 0:
            content_lines.append(lines[j])
            paren_count += lines[j].count('(') - lines[j].count(')')
            j += 1
        
        # Extract the content without asyncio.run wrapper
        full_content = ''.join(content_lines)
        # Remove "asyncio.run(" from the start
        content = full_content.replace("asyncio.run(", "", 1)
        # Remove the final closing paren
        content = content.rstrip()
        if content.endswith(")"):
            content = content[:-1]
        content += "\n"
        
        new_lines.append(content)
        skip_until = j
        continue
    
    new_lines.append(line)

with open("exp/kmeans.py", "w") as f:
    f.writelines(new_lines)

print("Final cleanup complete!")
