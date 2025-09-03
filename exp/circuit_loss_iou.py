"""Circuit loss using IoU."""

import torch as th


def compute_circuit_loss_iou(
    data: th.Tensor,
    circuit: th.Tensor,
) -> th.Tensor:
    """Compute circuit loss using IoU.

    Args:
        data: Boolean tensor of shape (batch_size, num_layers, num_experts)
        circuit: Circuit tensor of shape (num_layers, num_experts)

    Returns:
        Loss tensor of shape (batch_size,)
    """
    # Convert to float
    data = data.float()
    circuit = circuit.float()

    # Compute intersection and union
    intersection = th.sum(data * circuit.unsqueeze(0), dim=(1, 2))
    union = th.sum(th.clamp(data + circuit.unsqueeze(0), 0, 1), dim=(1, 2))

    # Compute IoU
    iou = intersection / (union + 1e-8)
    return iou

