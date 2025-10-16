"""Tests to ensure code quality and best practices in production code."""

import ast
from pathlib import Path


def get_production_python_files() -> set[Path]:
    """Get all Python files in production code (excluding test directories)."""
    production_files = set()

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Directories to scan for production code
    production_dirs = ["core", "exp", "viz"]

    for dir_name in production_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            # Find all .py files recursively
            for py_file in dir_path.rglob("*.py"):
                # Skip __pycache__ and other non-source files
                if "__pycache__" not in str(py_file):
                    production_files.add(py_file)

    return production_files


def extract_imports_from_file(file_path: Path) -> set[str]:
    """Extract all import statements from a Python file."""
    imports = set()

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
                # Also add the full import path for from imports
                for alias in node.names:
                    imports.add(f"{node.module}.{alias.name}")

    except (SyntaxError, UnicodeDecodeError) as e:
        # If we can't parse the file, skip it but print a warning
        print(f"Warning: Could not parse {file_path}: {e}")

    return imports


def test_no_mock_imports_in_production_code():
    """Test that production code does not import any mocking modules."""

    # Mock-related modules that should not be imported in production code
    forbidden_mock_imports = {
        "unittest.mock",
        "mock",
        "pytest.mock",
        "unittest.mock.Mock",
        "unittest.mock.MagicMock",
        "unittest.mock.patch",
        "unittest.mock.AsyncMock",
        "mock.Mock",
        "mock.MagicMock",
        "mock.patch",
    }

    production_files = get_production_python_files()
    violations = []

    for file_path in production_files:
        imports = extract_imports_from_file(file_path)

        # Check for forbidden imports
        violations.extend(
            (file_path, import_name)
            for import_name in imports
            if any(forbidden in import_name for forbidden in forbidden_mock_imports)
        )

    # Assert no violations found
    if violations:
        violation_messages = []
        for file_path, import_name in violations:
            relative_path = file_path.relative_to(Path(__file__).parent.parent)
            violation_messages.append(f"  {relative_path}: imports '{import_name}'")

        error_message = (
            "Found mock imports in production code:\n"
            + "\n".join(violation_messages)
            + "\n\nProduction code should not import mocking modules. "
            "Use duck typing or capability checking instead."
        )

        raise AssertionError(error_message)

    # If we get here, no violations were found
    print(
        f"✅ Checked {len(production_files)} production files - no mock imports found"
    )


def test_production_files_exist():
    """Sanity check that we're actually finding production files to test."""
    production_files = get_production_python_files()

    # We should find at least some files
    assert len(production_files) > 0, "No production Python files found to check"

    # Check that we found files in expected directories
    found_dirs = set()
    for file_path in production_files:
        # Get the first directory component relative to project root
        relative_path = file_path.relative_to(Path(__file__).parent.parent)
        first_dir = relative_path.parts[0]
        found_dirs.add(first_dir)

    expected_dirs = {"core", "exp", "viz"}
    found_expected = found_dirs.intersection(expected_dirs)

    assert len(found_expected) > 0, (
        f"Expected to find files in {expected_dirs}, but only found files in {found_dirs}"
    )

    print(
        f"✅ Found {len(production_files)} production files in directories: {sorted(found_dirs)}"
    )


if __name__ == "__main__":
    # Allow running this test directly
    test_production_files_exist()
    test_no_mock_imports_in_production_code()
    print("All code quality tests passed!")
