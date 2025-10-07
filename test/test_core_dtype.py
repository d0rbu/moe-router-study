"""Tests for core.dtype module."""

import pytest
import torch as th

from core.dtype import DTYPE_ALIASES, DTYPE_MAP, get_dtype


class TestDtypeAliases:
    """Test dtype alias mappings."""

    def test_dtype_aliases_structure(self):
        """Test that DTYPE_ALIASES has expected structure."""
        assert isinstance(DTYPE_ALIASES, dict)
        assert len(DTYPE_ALIASES) > 0

        for dtype, aliases in DTYPE_ALIASES.items():
            assert isinstance(dtype, th.dtype)
            assert isinstance(aliases, set)
            assert len(aliases) > 0
            assert all(isinstance(alias, str) for alias in aliases)

    def test_dtype_map_structure(self):
        """Test that DTYPE_MAP has expected structure."""
        assert isinstance(DTYPE_MAP, dict)
        assert len(DTYPE_MAP) > 0

        for alias, dtype in DTYPE_MAP.items():
            assert isinstance(alias, str)
            assert isinstance(dtype, th.dtype)

    def test_dtype_map_consistency(self):
        """Test that DTYPE_MAP is consistent with DTYPE_ALIASES."""
        # Every alias in DTYPE_ALIASES should be in DTYPE_MAP
        expected_aliases = set()
        for aliases in DTYPE_ALIASES.values():
            expected_aliases.update(aliases)

        assert set(DTYPE_MAP.keys()) == expected_aliases

        # Every mapping should be correct
        for dtype, aliases in DTYPE_ALIASES.items():
            for alias in aliases:
                assert DTYPE_MAP[alias] == dtype


class TestGetDtype:
    """Test get_dtype function."""

    @pytest.mark.parametrize(
        "dtype_str,expected_dtype",
        [
            ("float32", th.float32),
            ("float", th.float32),
            ("fp32", th.float32),
            ("f32", th.float32),
            ("float16", th.float16),
            ("fp16", th.float16),
            ("f16", th.float16),
            ("bfloat16", th.bfloat16),
            ("bf16", th.bfloat16),
            ("float64", th.float64),
            ("fp64", th.float64),
            ("f64", th.float64),
        ],
    )
    def test_get_dtype_valid_inputs(self, dtype_str: str, expected_dtype: th.dtype):
        """Test get_dtype with valid inputs."""
        result = get_dtype(dtype_str)
        assert result == expected_dtype

    def test_get_dtype_invalid_input(self):
        """Test get_dtype with invalid input."""
        with pytest.raises(ValueError, match="Invalid dtype: invalid_dtype"):
            get_dtype("invalid_dtype")

    def test_get_dtype_empty_string(self):
        """Test get_dtype with empty string."""
        with pytest.raises(ValueError, match="Invalid dtype: "):
            get_dtype("")

    def test_get_dtype_none_input(self):
        """Test get_dtype with None input."""
        with pytest.raises((ValueError, TypeError)):
            get_dtype(None)

    def test_get_dtype_case_sensitivity(self):
        """Test that get_dtype is case sensitive."""
        # Should work with lowercase
        assert get_dtype("float32") == th.float32

        # Should fail with uppercase (assuming aliases are lowercase)
        with pytest.raises(ValueError):
            get_dtype("FLOAT32")

    def test_get_dtype_all_aliases(self):
        """Test get_dtype with all defined aliases."""
        for alias in DTYPE_MAP:
            result = get_dtype(alias)
            assert isinstance(result, th.dtype)
            assert result == DTYPE_MAP[alias]


class TestDtypeAliasesContent:
    """Test specific content of dtype aliases."""

    def test_float32_aliases(self):
        """Test float32 aliases."""
        expected_aliases = {"float32", "float", "fp32", "f32"}
        assert th.float32 in DTYPE_ALIASES
        assert DTYPE_ALIASES[th.float32] == expected_aliases

    def test_float16_aliases(self):
        """Test float16 aliases."""
        expected_aliases = {"float16", "fp16", "f16"}
        assert th.float16 in DTYPE_ALIASES
        # Note: there's a duplicate in the original code, but the set should deduplicate
        assert expected_aliases.issubset(DTYPE_ALIASES[th.float16])

    def test_bfloat16_aliases(self):
        """Test bfloat16 aliases."""
        expected_aliases = {"bfloat16", "bf16"}
        assert th.bfloat16 in DTYPE_ALIASES
        assert DTYPE_ALIASES[th.bfloat16] == expected_aliases

    def test_float64_aliases(self):
        """Test float64 aliases."""
        expected_aliases = {"float64", "fp64", "f64"}
        assert th.float64 in DTYPE_ALIASES
        assert DTYPE_ALIASES[th.float64] == expected_aliases


class TestDtypeIntegration:
    """Integration tests for dtype functionality."""

    def test_dtype_roundtrip(self):
        """Test that we can convert dtype to string and back."""
        test_dtypes = [th.float32, th.float16, th.bfloat16, th.float64]

        for dtype in test_dtypes:
            # Get an alias for this dtype
            aliases = DTYPE_ALIASES[dtype]
            alias = next(iter(aliases))  # Get first alias

            # Convert back
            result = get_dtype(alias)
            assert result == dtype

    def test_tensor_creation_with_get_dtype(self):
        """Test creating tensors with dtypes from get_dtype."""
        for alias in ["float32", "float16", "bfloat16"]:
            dtype = get_dtype(alias)
            tensor = th.zeros(2, 3, dtype=dtype)
            assert tensor.dtype == dtype
            assert tensor.shape == (2, 3)
