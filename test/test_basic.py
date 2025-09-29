"""Basic test to ensure pytest can run successfully."""


def test_basic():
    """Basic test that always passes."""
    assert True


def test_imports():
    """Test that main modules can be imported."""
    import core  # noqa: F401
    import exp  # noqa: F401
    import viz  # noqa: F401

    assert True
