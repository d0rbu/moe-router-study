import torch as th


def max_iou_and_index(
    data: th.Tensor, circuits: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    assert data.ndim == 3, "data must be of shape (batch_size, num_layers, num_experts)"
    assert circuits.ndim == 3, (
        "circuits must be of shape (num_circuits, num_layers, num_experts)"
    )

    batch_size, num_layers, num_experts = data.shape
    num_circuits = circuits.shape[0]
    assert circuits.shape[1] == num_layers, (
        "circuits must have the same number of layers as data"
    )
    assert circuits.shape[2] == num_experts, (
        "circuits must have the same number of experts as data"
    )
    assert circuits.dtype == th.bool, "circuits must be a boolean tensor"

    # compute the max IoU for each data point
    data_flat = data.view(batch_size, 1, -1)
    circuits_flat = circuits.view(1, num_circuits, -1)
    intersection = th.sum(data_flat & circuits_flat, dim=-1)
    union = th.sum(data_flat | circuits_flat, dim=-1)
    iou = intersection / union

    assert iou.dtype == th.float32, "iou must be a float32 tensor"
    assert iou.shape == (batch_size, num_circuits), (
        "iou must be of shape (batch_size, num_circuits)"
    )

    max_iou, max_iou_idx = th.max(iou, dim=1)

    return max_iou, max_iou_idx


def mean_max_iou(data: th.Tensor, circuits: th.Tensor) -> th.Tensor:
    max_iou, max_iou_idx = max_iou_and_index(data, circuits)
    return max_iou.mean()


def circuit_score(
    data: th.Tensor,
    circuits: th.Tensor,
    complexity_coefficient: float = 1.0,
    complexity_power: float = 1.0,
) -> th.Tensor:
    iou = mean_max_iou(data, circuits)
    complexity = circuits.sum()

    return iou - complexity_coefficient * complexity.pow(complexity_power)
