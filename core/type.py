from typing import Any, cast


def assert_type[T](obj: Any, typ: type[T]) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast("T", obj)
