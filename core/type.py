import types
from typing import Any, cast


def assert_type[T](obj: Any, typ: type[T]) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        # Handle union types (both old Union and new | syntax)
        if hasattr(typ, "__name__"):
            type_name = typ.__name__
        elif isinstance(typ, types.UnionType):
            # For X | Y syntax, get a readable representation
            type_name = str(typ)
        else:
            # Fallback for other complex types
            type_name = str(typ)

        raise TypeError(f"Expected {type_name}, got {type(obj).__name__}")

    return cast("T", obj)
