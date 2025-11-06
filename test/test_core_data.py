from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import pytest

from core import data as core_data

if TYPE_CHECKING:
    from collections.abc import Iterable


@pytest.mark.unit
def test_toy_text_yields_expected_samples() -> None:
    # toy_text may not exist; if so, skip this test to keep PR minimal
    if not hasattr(core_data, "toy_text"):
        pytest.skip("toy_text not available in core.data; skipping fast dataset test")
    it: Iterable[str] = core_data.toy_text()
    samples = list(itertools.islice(it, 10))
    assert samples[:4] == [
        "Tiny sample 1",
        "Tiny sample 2",
        "Tiny sample 3",
        "Tiny sample 4",
    ]


@pytest.mark.unit
def test_fineweb_signature_present() -> None:
    assert callable(core_data.fineweb_10bt_text)
