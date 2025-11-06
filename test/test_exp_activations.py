from pathlib import Path

import pytest
import torch as th

from exp import activations as act


@pytest.mark.unit
def test_load_activations_indices_tokens_and_topk_from_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: create a fake router_logits directory with two tiny batches
    router_dir = tmp_path / "router_logits"
    router_dir.mkdir(parents=True, exist_ok=True)

    # Two files: 0.pt and 1.pt
    def write_file(
        idx: int, batch: int = 2, layers: int = 3, experts: int = 4, topk: int = 2
    ) -> None:
        # random logits -> doesn't matter as long as shape (B, L, E)
        logits = th.randn(batch, layers, experts)
        tokens = [[f"t{idx}-{bi}-{li}" for li in range(layers)] for bi in range(batch)]
        th.save(
            {"router_logits": logits, "tokens": tokens, "topk": topk},
            router_dir / f"{idx}.pt",
        )

    write_file(0)
    write_file(1)

    # Monkeypatch the directory constant directly
    monkeypatch.setattr("exp.get_router_activations.ROUTER_LOGITS_DIR", str(router_dir))

    # Act
    activated_experts, indices, tokens, top_k = (
        act.load_activations_indices_tokens_and_topk(device="cpu")
    )

    # Assert
    # Shape checks
    assert activated_experts.dtype == th.bool
    assert activated_experts.shape[:2] == (4, 3)  # 2 files * 2 batch, 3 layers
    assert activated_experts.shape[2] == 4  # experts

    assert indices.dtype == th.long
    assert indices.shape == (4, 3, top_k)

    assert len(tokens) == 4  # two files * two batch
    assert top_k == 2

    # Activated mask must contain exactly top_k True along expert dim
    trues_per_token = activated_experts.sum(dim=2)
    assert th.all(trues_per_token == top_k)


@pytest.mark.unit
def test_load_activations_raises_on_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch to a non-existent directory with no 0.pt file
    monkeypatch.setattr(
        "exp.get_router_activations.ROUTER_LOGITS_DIR", str(tmp_path / "missing")
    )

    with pytest.raises(ValueError):
        act.load_activations_and_topk(device="cpu")
