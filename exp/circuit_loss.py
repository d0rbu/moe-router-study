import torch as th
import torch.nn.functional as F


def max_iou_and_index(
    data: th.Tensor, circuits: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    assert data.ndim == 3, "data must be of shape (batch_size, num_layers, num_experts)"
    assert circuits.ndim >= 3, (
        "circuits must be of shape (*, num_circuits, num_layers, num_experts)"
    )

    batch_size, num_layers, num_experts = data.shape
    num_circuits = circuits.shape[-3]
    assert circuits.shape[-2] == num_layers, (
        "circuits must have the same number of layers as data"
    )
    assert circuits.shape[-1] == num_experts, (
        "circuits must have the same number of experts as data"
    )
    assert circuits.dtype == th.bool, "circuits must be a boolean tensor"

    num_extra_dims = circuits.ndim - 3

    # compute the max IoU for each data point
    data_flat = data.view(
        *([1] * num_extra_dims), batch_size, 1, num_layers * num_experts
    )
    circuits_flat = circuits.view(
        *circuits.shape[:-3], 1, num_circuits, num_layers * num_experts
    )
    intersection = th.sum(data_flat & circuits_flat, dim=-1)
    union = th.sum(data_flat | circuits_flat, dim=-1)
    iou = intersection / union

    assert iou.dtype == th.float32, "iou must be a float32 tensor"
    assert iou.shape[-1] == num_circuits, (
        "iou must have the same number of circuits as data"
    )
    assert iou.shape[-2] == batch_size, (
        "iou must have the same number of data points as data"
    )

    # (..., B, C) -> (..., B)
    max_iou, max_iou_idx = th.max(iou, dim=-1)

    return max_iou, max_iou_idx


def mean_max_iou(data: th.Tensor, circuits: th.Tensor) -> th.Tensor:
    max_iou, _max_iou_idx = max_iou_and_index(data, circuits)

    # (..., B) -> (...)
    return max_iou.mean(dim=-1)


def hard_circuit_score(
    data: th.Tensor,
    circuits: th.Tensor,
    complexity_coefficient: float = 1.0,
    complexity_power: float = 1.0,
) -> th.Tensor:
    iou = mean_max_iou(data, circuits)
    # (..., C, L, E) -> (...)
    complexity = circuits.sum(dim=(-3, -2, -1))

    return iou - complexity_coefficient * complexity.pow(complexity_power)


def min_logit_loss_and_index(
    data: th.Tensor,
    circuits_logits: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    assert data.ndim == 3, "data must be of shape (batch_size, num_layers, num_experts)"
    assert circuits_logits.ndim >= 3, (
        "circuits_logits must be of shape (*, num_circuits, num_layers, num_experts)"
    )

    batch_size, num_layers, num_experts = data.shape
    num_circuits = circuits_logits.shape[-3]
    assert circuits_logits.shape[-2] == num_layers, (
        "circuits_logits must have the same number of layers as data"
    )
    assert circuits_logits.shape[-1] == num_experts, (
        "circuits_logits must have the same number of experts as data"
    )
    assert circuits_logits.dtype == th.float32, (
        "circuits_logits must be a float32 tensor"
    )

    num_extra_dims = circuits_logits.ndim - 3

    data = data.float()
    circuits = th.sigmoid(circuits_logits)

    # get the closest circuit for each data point
    data_flat = data.view(*([1] * num_extra_dims), batch_size, num_layers * num_experts)
    circuits_flat = circuits.view(
        *circuits_logits.shape[:-3], num_circuits, num_layers * num_experts
    )

    # (..., B, C)
    circuit_distances = th.cdist(data_flat, circuits_flat, p=1)

    # (..., B, C) -> (..., B)
    _min_distance, min_distance_idx = circuit_distances.min(dim=-1)

    # (..., C, L, E) -> (..., B, L, E)
    closest_circuit_logits = circuits_logits[min_distance_idx]

    # (..., B, L, E) -> (..., B)
    bce_loss = F.binary_cross_entropy_with_logits(
        closest_circuit_logits, data, reduction="none"
    ).sum(dim=(-2, -1))

    return bce_loss, min_distance_idx


def min_logit_loss(
    data: th.Tensor,
    circuits_logits: th.Tensor,
) -> th.Tensor:
    min_loss, _min_loss_idx = min_logit_loss_and_index(data, circuits_logits)

    # (..., B) -> (...)
    return min_loss.mean(dim=-1)


def circuit_loss(
    data: th.Tensor,
    circuits_logits: th.Tensor,
    top_k: int,
    complexity_importance: float = 1.0,
    complexity_power: float = 1.0,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    faithfulness_loss = min_logit_loss(data, circuits_logits)

    # (..., C, L, E)
    complexity = th.sigmoid(circuits_logits)
    # sum over experts and account for top-k
    # (..., C, L, E) -> (..., C, L)
    complexity = complexity.sum(dim=-1) / top_k
    # average over layers
    # (..., C, L) -> (..., C)
    complexity = complexity.mean(dim=-1)
    # sum over circuits
    # (..., C) -> (...)
    complexity = complexity.sum(dim=-1)

    assert complexity_importance >= 0.0 and complexity_importance <= 1.0, (
        "complexity_importance must be between 0.0 and 1.0"
    )

    faithfulness_importance = 1.0 - complexity_importance

    loss = (
        faithfulness_importance * faithfulness_loss
        + complexity_importance * complexity.pow(complexity_power)
    )

    return loss, faithfulness_loss, complexity
