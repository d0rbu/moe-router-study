"""Basic sanity imports test for this project."""


def test_imports() -> None:
    """Only verify that core modules import and expose expected constants."""
    import core
    import exp
    import viz

    assert core.__version__ == "0.1.0"
    assert hasattr(exp, "BASE_OUTPUT_DIR")
    assert hasattr(viz, "BASE_FIGURE_DIR")

