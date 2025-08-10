"""Tests for core.data module (toy dataset only)."""

from core.data import DATASETS


def test_datasets_registry_has_toy() -> None:
    assert "toy" in DATASETS
    func = DATASETS["toy"]
    samples = list(func())
    assert samples == [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]


def test_datasets_registry_is_well_formed() -> None:
    assert isinstance(DATASETS, dict)
    for k, v in DATASETS.items():
        assert isinstance(k, str)
        assert callable(v)
