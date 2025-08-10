# Normalize by theoretical max: top_k * num_layers
_, top_k = load_activations_and_topk(device=device)
L = circuits.shape[-2]
denom_val = float(top_k * L) if L is not None else 1.0
denom = th.tensor(denom_val, device=activations.device, dtype=activations.dtype)
norm_scores = (activations / denom).clamp(0, 1)
