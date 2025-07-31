"""Basic tests to verify the setup works."""

import pytest


def test_imports() -> None:
    """Test that basic imports work."""
    import core

    assert core.__version__ == "0.1.0"


def test_basic_functionality() -> None:
    """Test basic functionality."""
    assert 1 + 1 == 2


@pytest.mark.slow
def test_nnterp_import() -> None:
    """Test that nnterp can be imported (marked as slow since it's a heavy import)."""
<<<<<<< HEAD
    try:
        import nnterp

        assert hasattr(nnterp, "StandardizedTransformer")
    except ImportError:
        pytest.skip("nnterp not installed or not available")
=======
    import nnterp
    assert hasattr(nnterp, 'StandardizedTransformer')
>>>>>>> f21a0d2 (Address all feedback: switch to ty, Python 3.12 only, run CI on all branches)
