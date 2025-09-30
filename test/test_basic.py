"""Basic tests to ensure the project structure is valid."""

import pytest


def test_imports():
    """Test that core modules can be imported."""
    import importlib.util

    modules = ["core", "exp", "viz"]
    for module_name in modules:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            pytest.fail(f"Module {module_name} not found")

        # Actually import to test for import errors
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
