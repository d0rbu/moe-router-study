"""Tests for core.data module (CI-friendly, no external dataset loads)."""

from collections.abc import Iterable
import inspect

from datasets import IterableColumn

from core.data import DATASETS, toy_text


class TestToyText:
    """Tests for toy_text function."""

    def test_toy_text_basic(self):
        text_column = toy_text()
        assert list(text_column) == [
            "Tiny sample 1",
            "Tiny sample 2",
            "Tiny sample 3",
            "Tiny sample 4",
        ]

    def test_toy_text_is_iterable(self):
        text_column = toy_text()
        assert isinstance(text_column, Iterable)

    def test_toy_text_type_annotation(self):
        from core.data import toy_text as fn

        sig = inspect.signature(fn)
        assert sig.return_annotation == IterableColumn


class TestDatasetsRegistry:
    """Tests for the DATASETS registry."""

    def test_datasets_registry_structure(self):
        assert isinstance(DATASETS, dict)
        assert len(DATASETS) > 0
        for name, func in DATASETS.items():
            assert isinstance(name, str)
            assert callable(func)

    def test_toy_in_registry(self):
        assert "toy" in DATASETS
        assert callable(DATASETS["toy"])  # function exists and is callable


class TestDatasetIntegration:
    """Lightweight integration-like checks using the toy dataset."""

    def test_dataset_iteration_pattern(self):
        text_column = toy_text()
        collected_texts = list(text_column)
        assert collected_texts == [
            "Tiny sample 1",
            "Tiny sample 2",
            "Tiny sample 3",
            "Tiny sample 4",
        ]

    def test_multiple_dataset_calls_consistency(self):
        result1 = list(toy_text())
        result2 = list(toy_text())
        assert (
            result1
            == result2
            == [
                "Tiny sample 1",
                "Tiny sample 2",
                "Tiny sample 3",
                "Tiny sample 4",
            ]
        )
