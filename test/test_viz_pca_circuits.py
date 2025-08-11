import os
from pathlib import Path

import pytest
import torch as th

import viz.pca_circuits as pca_mod


@pytest.mark.unit
def test_pca_figure_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: monkeypatch load_activations_and_topk to return small boolean tensor and top_k
    def fake_load_activations_and_topk(device: str = "cpu"):
        # (B,L,E) small example: B=3, L=2, E=4, top_k=1
        activated = th.zeros(3, 2, 4, dtype=th.bool)
        activated[:, :, 0] = True
        return activated, 1

    monkeypatch.setattr(pca_mod.act, "load_activations_and_topk", fake_load_activations_and_topk)

    # Patch FIGURE_DIR to tmp by editing pca_mod.FIGURE_PATH
    monkeypatch.setattr(pca_mod, "FIGURE_PATH", os.path.join(str(tmp_path), "pca_circuits.png"))

    # Act
    pca_mod.pca_figure()

    # Assert
    assert (tmp_path / "pca_circuits.png").exists()
