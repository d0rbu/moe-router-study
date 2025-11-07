import re

with open("exp/kmeans.py", "r") as f:
    content = f.read()

# Pattern to match safe_await_with_worker_check calls
# They look like: safe_await_with_worker_check(actual_call(...), workers=..., ...)
# We want to replace with just: actual_call(...)

# First pattern: multi-line safe_await_with_worker_check
pattern1 = r'safe_await_with_worker_check\(\s*(\w+)\((.*?)\),\s*workers=.*?\)'
replacement1 = r'\1(\2)'
content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# Write back
with open("exp/kmeans.py", "w") as f:
    f.write(content)

print("Removed safe_await_with_worker_check calls!")
