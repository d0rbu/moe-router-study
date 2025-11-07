with open("exp/kmeans.py", "r") as f:
    content = f.read()

# Fix the broken load_activations_and_init_dist call
broken_pattern = """activations, activation_dims, gpu_process_group, gpu_process_groups = load_activations_and_init_dist(
    activations, activation_dims, gpu_process_group, gpu_process_groups = load_activations_and_init_dist(
        load_activations_and_init_dist(
            model_name=model_name,
            dataset_name=dataset_name,
            tokens_per_file=tokens_per_file,
            reshuffled_tokens_per_file=reshuffled_tokens_per_file,
            submodule_names=[ActivationKeys.ROUTER_LOGITS, ActivationKeys.MLP_OUTPUT],
            context_length=context_length,
            num_workers=num_workers,
            debug=log_level_numeric <= debug_level_numeric,
            device_type=device_type,
        )"""

fixed_pattern = """activations, activation_dims, gpu_process_group, gpu_process_groups = load_activations_and_init_dist(
        model_name=model_name,
        dataset_name=dataset_name,
        tokens_per_file=tokens_per_file,
        reshuffled_tokens_per_file=reshuffled_tokens_per_file,
        submodule_names=[ActivationKeys.ROUTER_LOGITS, ActivationKeys.MLP_OUTPUT],
        context_length=context_length,
        num_workers=num_workers,
        debug=log_level_numeric <= debug_level_numeric,
        device_type=device_type,
    )"""

if broken_pattern in content:
    content = content.replace(broken_pattern, fixed_pattern)
else:
    print("Pattern not found, trying alternative fix...")
    # Alternative: find and fix using regex
    import re
    # Find the assignment and fix it
    pattern = r'activations, activation_dims, gpu_process_group, gpu_process_groups = \n\s+load_activations_and_init_dist\('
    replacement = 'activations, activation_dims, gpu_process_group, gpu_process_groups = load_activations_and_init_dist('
    content = re.sub(pattern, replacement, content)

with open("exp/kmeans.py", "w") as f:
    f.write(content)

print("Fixed asyncio.run removal!")
