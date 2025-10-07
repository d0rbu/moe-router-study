"""Tests for core.type module."""

import pytest

from core.type import assert_type


class TestAssertType:
    """Test assert_type function."""
    
    def test_assert_type_correct_type(self):
        """Test assert_type with correct type."""
        # Test with various types
        result = assert_type("hello", str)
        assert result == "hello"
        assert isinstance(result, str)
        
        result = assert_type(42, int)
        assert result == 42
        assert isinstance(result, int)
        
        result = assert_type([1, 2, 3], list)
        assert result == [1, 2, 3]
        assert isinstance(result, list)
        
        result = assert_type({"key": "value"}, dict)
        assert result == {"key": "value"}
        assert isinstance(result, dict)
    
    def test_assert_type_wrong_type(self):
        """Test assert_type with wrong type."""
        with pytest.raises(TypeError, match="Expected str, got int"):
            assert_type(42, str)
        
        with pytest.raises(TypeError, match="Expected int, got str"):
            assert_type("hello", int)
        
        with pytest.raises(TypeError, match="Expected list, got dict"):
            assert_type({"key": "value"}, list)
        
        with pytest.raises(TypeError, match="Expected dict, got list"):
            assert_type([1, 2, 3], dict)
    
    def test_assert_type_none_value(self):
        """Test assert_type with None value."""
        # None should fail for non-None types
        with pytest.raises(TypeError, match="Expected str, got NoneType"):
            assert_type(None, str)
        
        with pytest.raises(TypeError, match="Expected int, got NoneType"):
            assert_type(None, int)
        
        # But should work for type(None)
        result = assert_type(None, type(None))
        assert result is None
    
    def test_assert_type_inheritance(self):
        """Test assert_type with inheritance."""
        class Parent:
            pass
        
        class Child(Parent):
            pass
        
        child_instance = Child()
        
        # Child should be accepted as Parent
        result = assert_type(child_instance, Parent)
        assert result is child_instance
        assert isinstance(result, Parent)
        assert isinstance(result, Child)
        
        # Child should be accepted as Child
        result = assert_type(child_instance, Child)
        assert result is child_instance
        assert isinstance(result, Child)
        
        # Parent should not be accepted as Child
        parent_instance = Parent()
        with pytest.raises(TypeError, match="Expected Child, got Parent"):
            assert_type(parent_instance, Child)
    
    def test_assert_type_builtin_inheritance(self):
        """Test assert_type with built-in type inheritance."""
        # bool is a subclass of int
        result = assert_type(True, int)
        assert result is True
        assert isinstance(result, bool)
        assert isinstance(result, int)
        
        # But int is not a bool
        with pytest.raises(TypeError, match="Expected bool, got int"):
            assert_type(42, bool)
    
    def test_assert_type_complex_types(self):
        """Test assert_type with complex types."""
        import collections.abc
        
        # Test with abstract base classes
        result = assert_type([1, 2, 3], collections.abc.Sequence)
        assert result == [1, 2, 3]
        
        result = assert_type("hello", collections.abc.Sequence)
        assert result == "hello"
        
        result = assert_type({"a": 1, "b": 2}, collections.abc.Mapping)
        assert result == {"a": 1, "b": 2}
    
    def test_assert_type_return_value_identity(self):
        """Test that assert_type returns the same object."""
        original_list = [1, 2, 3]
        result = assert_type(original_list, list)
        assert result is original_list  # Same object, not a copy
        
        original_dict = {"key": "value"}
        result = assert_type(original_dict, dict)
        assert result is original_dict  # Same object, not a copy
    
    def test_assert_type_with_custom_classes(self):
        """Test assert_type with custom classes."""
        class CustomClass:
            def __init__(self, value):
                self.value = value
            
            def __eq__(self, other):
                return isinstance(other, CustomClass) and self.value == other.value
        
        instance = CustomClass(42)
        result = assert_type(instance, CustomClass)
        assert result is instance
        assert result.value == 42
        
        # Wrong type should fail
        with pytest.raises(TypeError, match="Expected CustomClass, got str"):
            assert_type("not a custom class", CustomClass)
    
    def test_assert_type_edge_cases(self):
        """Test assert_type with edge cases."""
        # Empty containers
        result = assert_type([], list)
        assert result == []
        
        result = assert_type({}, dict)
        assert result == {}
        
        result = assert_type("", str)
        assert result == ""
        
        # Zero values
        result = assert_type(0, int)
        assert result == 0
        
        result = assert_type(0.0, float)
        assert result == 0.0
    
    def test_assert_type_generic_types(self):
        """Test assert_type behavior with generic types."""
        # Note: assert_type doesn't handle generic types like List[int]
        # It only works with the base types
        from typing import List, Dict
        
        # This should work (checking against the base type)
        result = assert_type([1, 2, 3], list)
        assert result == [1, 2, 3]
        
        # Generic types from typing module are not directly supported
        # This is expected behavior for this simple implementation
        with pytest.raises(TypeError):
            assert_type([1, 2, 3], List[int])


class TestAssertTypeErrorMessages:
    """Test error messages from assert_type."""
    
    def test_error_message_format(self):
        """Test that error messages have the expected format."""
        try:
            assert_type(42, str)
        except TypeError as e:
            error_msg = str(e)
            assert "Expected str, got int" in error_msg
        
        try:
            assert_type("hello", int)
        except TypeError as e:
            error_msg = str(e)
            assert "Expected int, got str" in error_msg
    
    def test_error_message_with_custom_class(self):
        """Test error messages with custom classes."""
        class MyCustomClass:
            pass
        
        try:
            assert_type("not my class", MyCustomClass)
        except TypeError as e:
            error_msg = str(e)
            assert "Expected MyCustomClass, got str" in error_msg
