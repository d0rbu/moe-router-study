import torch as th
import torch.nn.functional as F


# New: factor IoU computation into a helper for direct testing
def compute_iou(data: th.Tensor, circuits: th.Tensor) -> th.Tensor:
    """Compute IoU between boolean data masks and circuit masks.

    Args:
        data: (B, L, E) boolean tensor
        circuits: (..., C, L, E) boolean tensor

    Returns:
        iou: (..., B, C) float32 tensor with IoU scores per data item and circuit
    """
    assert data.ndim == 3, "data must be of shape (batch_size, num_layers, num_experts)"
    assert circuits.ndim >= 3, (
        "circuits must be of shape (*, num_circuits, num_layers, num_experts)"
    )

    # Check that L and E dimensions match
    assert data.shape[1:] == circuits.shape[-2:], (
        f"Layer and expert dimensions must match: {data.shape[1:]} vs {circuits.shape[-2:]}"
    )

    # Get dimensions
    batch_size, num_layers, num_experts = data.shape
    circuits_shape = circuits.shape
    num_circuits = circuits_shape[-3]

    # Reshape data for broadcasting: (B, 1, L, E)
    data_reshaped = data.unsqueeze(1)

    # Reshape circuits for broadcasting: (..., C, L, E)
    circuits_reshaped = circuits

    # Compute intersection and union
    intersection = (data_reshaped & circuits_reshaped).sum(dim=(-2, -1))
    union = (data_reshaped | circuits_reshaped).sum(dim=(-2, -1))

    # Compute IoU
    iou = intersection.float() / union.float().clamp(min=1)

    # Reshape to (..., B, C)
    output_shape = (*circuits_shape[:-3], batch_size, num_circuits)
    return iou.view(*output_shape)


def compute_circuit_loss(
    data: th.Tensor,
    circuits_logits: th.Tensor,
    topk: int | None = None,
    top_k: int | None = None,
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Compute overall loss with faithfulness and complexity components.

    Accepts both `topk` and `top_k` for compatibility with tests.
    """
    # Normalize top-k argument
    if topk is None and top_k is None:
        raise TypeError("circuit_loss requires `topk` (or `top_k`) argument")
    if topk is None:
        topk = int(top_k)  # type: ignore[arg-type]
    elif top_k is not None and int(top_k) != int(topk):
        raise ValueError("Conflicting values for topk and top_k")

    faithfulness_loss = min_logit_loss(data, circuits_logits)
    complexity_loss = complexity_importance * complexity_loss_fn(
        circuits_logits, topk, power=complexity_power
    )
    total_loss = faithfulness_loss + complexity_loss
    return total_loss, faithfulness_loss, complexity_loss


def min_logit_loss(
    data: th.Tensor, circuits_logits: th.Tensor, eps: float = 1e-6  # eps is unused but kept for API compatibility
) -> th.Tensor:
    """Compute minimum logit loss.

    Args:
        data: (B, L, E) boolean tensor
        circuits_logits: (C, L, E) float32 tensor with circuit logits
        eps: Small constant for numerical stability (unused but kept for API compatibility)

    Returns:
        loss: (B,) float32 tensor with loss per data item
    """
    # Convert data to float
    data = data.float()

    # Compute dot product between data and each circuit
    # (B, L, E) @ (C, L, E) -> (B, C)
    dot_products = th.einsum("ble,cle->bc", data, circuits_logits)

    # Compute minimum logit loss
    min_logits, _ = dot_products.min(dim=1)
    loss = -min_logits

    return loss


def min_logit_loss_and_index(
    data: th.Tensor, circuits_logits: th.Tensor, eps: float = 1e-6  # eps is unused but kept for API compatibility
) -> tuple[th.Tensor, th.Tensor]:
    """Compute minimum logit loss and corresponding circuit index.

    Args:
        data: (B, L, E) boolean tensor
        circuits_logits: (C, L, E) float32 tensor with circuit logits
        eps: Small constant for numerical stability (unused but kept for API compatibility)

    Returns:
        loss: (B,) float32 tensor with loss per data item
        min_idx: (B,) int64 tensor with index of minimum logit per data item
    """
    # Convert data to float
    data = data.float()

    # Compute dot product between data and each circuit
    # (B, L, E) @ (C, L, E) -> (B, C)
    dot_products = th.einsum("ble,cle->bc", data, circuits_logits)

    # Compute minimum logit loss and index
    min_logits, min_idx = dot_products.min(dim=1)
    loss = -min_logits

    return loss, min_idx


def complexity_loss_fn(
    circuits_logits: th.Tensor, topk: int, power: float = 1.0
) -> th.Tensor:
    """Compute complexity loss.

    Args:
        circuits_logits: (C, L, E) float32 tensor with circuit logits
        topk: Number of top experts to consider
        power: Power to raise the complexity to

    Returns:
        loss: () float32 tensor with complexity loss
    """
    # Get dimensions
    num_circuits, num_layers, num_experts = circuits_logits.shape

    # Compute top-k mask
    _, topk_indices = th.topk(circuits_logits, k=topk, dim=-1)
    topk_mask = F.one_hot(topk_indices, num_experts).sum(dim=-2).bool()

    # Compute complexity
    complexity = topk_mask.float().sum() / (num_circuits * num_layers)

    # Apply power
    if power != 1.0:
        complexity = th.pow(complexity, power)

    return complexity


# Alias for backward compatibility with tests
def circuit_loss(
    data: th.Tensor,
    circuits_logits: th.Tensor,
    topk: int | None = None,
    top_k: int | None = None,
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Alias for compute_circuit_loss for backward compatibility."""
    return compute_circuit_loss(
        data, circuits_logits, topk, top_k, complexity_importance, complexity_power
    )


def hard_circuit_score(
    data: th.Tensor, circuits: th.Tensor, topk: int  # topk is unused but kept for API compatibility
) -> tuple[th.Tensor, th.Tensor]:
    """Compute hard circuit score.

    Args:
        data: (B, L, E) boolean tensor
        circuits: (C, L, E) boolean tensor
        topk: Number of top experts to consider (unused but kept for API compatibility)

    Returns:
        score: (B, C) float32 tensor with score per data item and circuit
        iou: (B, C) float32 tensor with IoU per data item and circuit
    """
    # Compute IoU
    iou = compute_iou(data, circuits)

    # Compute score
    score = iou

    return score, iou


def max_iou_and_index(
    data: th.Tensor, circuits: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """Compute maximum IoU and corresponding circuit index.

    Args:
        data: (B, L, E) boolean tensor
        circuits: (C, L, E) boolean tensor

    Returns:
        max_iou: (B,) float32 tensor with maximum IoU per data item
        max_idx: (B,) int64 tensor with index of maximum IoU per data item
    """
    # Compute IoU
    iou = compute_iou(data, circuits)

    # Compute maximum IoU and index
    max_iou, max_idx = iou.max(dim=1)

    return max_iou, max_idx


def mean_max_iou(data: th.Tensor, circuits: th.Tensor) -> th.Tensor:
    """Compute mean of maximum IoU.

    Args:
        data: (B, L, E) boolean tensor
        circuits: (C, L, E) boolean tensor

    Returns:
        mean_max_iou: () float32 tensor with mean of maximum IoU
    """
    # Compute maximum IoU
    max_iou, _ = max_iou_and_index(data, circuits)

    # Compute mean
    mean_max_iou = max_iou.mean()

    return mean_max_iou

