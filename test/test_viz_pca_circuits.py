"""Tests for viz.pca_circuits without mocking matplotlib."""

from pathlib import Path

import torch as th


def test_pca_figure_creates_file(tmp_path: Path, monkeypatch) -> None:
    """Run pca_figure on tiny activation data and assert output image exists."""
    # 1) Create tiny activation file the loader expects
    router_logits_dir = tmp_path / "router_logits"
    router_logits_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "topk": 2,
        # Small batch=10, layers=2, experts=6
        "router_logits": th.randn(10, 2, 6),
    }
    th.save(data, router_logits_dir / "0.pt")

    # 2) Send exp paths and figure dir to temp
    monkeypatch.setattr("exp.ROUTER_LOGITS_DIR", str(router_logits_dir), raising=False)

    fig_dir = tmp_path / "fig"
    monkeypatch.setattr("viz.FIGURE_DIR", str(fig_dir), raising=False)

    # 3) Call pca_figure on CPU
    from viz.pca_circuits import pca_figure, FIGURE_PATH

    pca_figure(device="cpu")

    # 4) Assert file was created and is non-empty
    out_path = Path(FIGURE_PATH)
    assert out_path.exists() and out_path.is_file()
    assert out_path.stat().st_size > 0

