"""Focused tests for compute_iou in exp.circuit_loss.

These avoid mocking external libraries and work on tiny boolean tensors.
"""

import pytest
import torch as th

from exp.circuit_loss import compute_iou


def test_compute_iou_basic_overlap() -> None:
    # data: (B=2, L=1, E=4)
    data = th.tensor(
        [
            [[True, False, True, False]],
            [[False, True, False, True]],
        ],
        dtype=th.bool,
    )

    # circuits: (C=2, L=1, E=4)
    circuits = th.tensor(
        [
            [[True, True, False, False]],  # circuit 0
            [[False, True, True, False]],  # circuit 1
        ],
        dtype=th.bool,
    )

    # IoU shape: (B, C) -> (2, 2)
    iou = compute_iou(data, circuits)
    assert iou.shape == (2, 2)

    # All pairs here have union of size 3 and intersection of size 1 -> IoU = 1/3
    expected = th.tensor([[1 / 3, 1 / 3], [1 / 3, 1 / 3]], dtype=th.float32)
    assert th.allclose(iou, expected, atol=1e-6)


def test_compute_iou_full_and_zero_overlap() -> None:
    data = th.tensor([[[True, False]]], dtype=th.bool)  # (1,1,2)
    circuits = th.tensor(
        [
            [[True, False]],  # full overlap -> IoU 1.0
            [[False, True]],  # zero overlap -> IoU 0.0
        ],
        dtype=th.bool,
    )
    iou = compute_iou(data, circuits)
    assert th.allclose(iou[0, 0], th.tensor(1.0))
    assert th.allclose(iou[0, 1], th.tensor(0.0))


def test_compute_iou_broadcast_extra_dims() -> None:
    # data: (B=2, L=1, E=2)
    data = th.tensor([[[True, False]], [[False, True]]], dtype=th.bool)
    # circuits: (G=3, C=2, L=1, E=2) with an extra grouping dim
    base_circuits = th.tensor(
        [
            [[True, False]],
            [[False, True]],
        ],
        dtype=th.bool,
    )  # (C=2,1,2)
    circuits = base_circuits.unsqueeze(0).repeat(3, 1, 1, 1)

    iou = compute_iou(data, circuits)
    assert iou.shape == (3, 2, 2)  # (G, B, C)


def test_compute_iou_shape_mismatch_raises() -> None:
    data = th.zeros(2, 2, 3, dtype=th.bool)
    circuits = th.zeros(4, 2, 4, dtype=th.bool)  # wrong E
    with pytest.raises(AssertionError):
        compute_iou(data, circuits)
