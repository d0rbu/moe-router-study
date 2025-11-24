"""Tests for k-means autointerpretability experiment."""

import asyncio
from pathlib import Path
import shutil
import tempfile

import pytest
import torch as th

from exp.kmeans_autointerp import (
    CentroidExplanation,
    CentroidExplanationCache,
    KMeansAutoInterp,
    ValidationExample,
    load_kmeans_centroids,
)


class TestCentroidExplanation:
    """Test CentroidExplanation dataclass."""

    def test_to_dict(self):
        explanation = CentroidExplanation(
            centroid_id=0,
            explanation="Test explanation",
            confidence=0.8,
            top_k_examples=["example 1", "example 2"],
            timestamp="2024-01-01T00:00:00",
            metadata={"layer": 0},
        )

        result = explanation.to_dict()
        assert isinstance(result, dict)
        assert result["centroid_id"] == 0
        assert result["explanation"] == "Test explanation"
        assert result["confidence"] == 0.8

    def test_from_dict(self):
        data = {
            "centroid_id": 1,
            "explanation": "Another explanation",
            "confidence": 0.9,
            "top_k_examples": ["ex1"],
            "timestamp": "2024-01-01T00:00:00",
            "metadata": {"layer": 1},
        }

        explanation = CentroidExplanation.from_dict(data)
        assert explanation.centroid_id == 1
        assert explanation.explanation == "Another explanation"
        assert explanation.confidence == 0.9


class TestCentroidExplanationCache:
    """Test CentroidExplanationCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cache_initialization(self, temp_cache_dir):
        cache = CentroidExplanationCache(temp_cache_dir)
        assert cache.cache_dir.exists()
        assert cache.cache_dir == Path(temp_cache_dir)

    def test_cache_set_and_get(self, temp_cache_dir):
        cache = CentroidExplanationCache(temp_cache_dir)

        explanation = CentroidExplanation(
            centroid_id=5,
            explanation="Test",
            confidence=0.7,
            top_k_examples=["ex1", "ex2"],
            timestamp="2024-01-01T00:00:00",
            metadata={},
        )

        # Set and get from memory
        cache.set(explanation)
        result = cache.get(5)

        assert result is not None
        assert result.centroid_id == 5
        assert result.explanation == "Test"
        assert result.confidence == 0.7

    def test_cache_persistence(self, temp_cache_dir):
        """Test that cache persists to disk."""
        explanation = CentroidExplanation(
            centroid_id=10,
            explanation="Persistent test",
            confidence=0.85,
            top_k_examples=["example"],
            timestamp="2024-01-01T00:00:00",
            metadata={"test": True},
        )

        # Create cache, set, and destroy
        cache1 = CentroidExplanationCache(temp_cache_dir)
        cache1.set(explanation)
        del cache1

        # Create new cache instance and check if data persists
        cache2 = CentroidExplanationCache(temp_cache_dir)
        result = cache2.get(10)

        assert result is not None
        assert result.centroid_id == 10
        assert result.explanation == "Persistent test"

    def test_cache_has(self, temp_cache_dir):
        cache = CentroidExplanationCache(temp_cache_dir)

        assert not cache.has(0)

        explanation = CentroidExplanation(
            centroid_id=0,
            explanation="Test",
            confidence=0.5,
            top_k_examples=[],
            timestamp="2024-01-01T00:00:00",
            metadata={},
        )
        cache.set(explanation)

        assert cache.has(0)
        assert not cache.has(1)

    def test_cache_clear(self, temp_cache_dir):
        cache = CentroidExplanationCache(temp_cache_dir)

        # Add multiple explanations
        for i in range(5):
            explanation = CentroidExplanation(
                centroid_id=i,
                explanation=f"Test {i}",
                confidence=0.5,
                top_k_examples=[],
                timestamp="2024-01-01T00:00:00",
                metadata={},
            )
            cache.set(explanation)

        assert cache.has(0)
        assert cache.has(4)

        # Clear cache
        cache.clear()

        assert not cache.has(0)
        assert not cache.has(4)


class TestKMeansAutoInterp:
    """Test KMeansAutoInterp class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.config.hidden_size = 64
        tokenizer = MagicMock()
        tokenizer.convert_ids_to_tokens.return_value = ["test", "tokens"]
        tokenizer.decode.return_value = "test tokens"
        model.tokenizer = tokenizer
        return model

    @pytest.fixture
    def dummy_centroids(self):
        """Create dummy centroids for testing."""
        return th.randn(10, 64)  # 10 centroids, 64 dimensions

    def test_initialization(self, temp_cache_dir, mock_model, dummy_centroids):
        """Test KMeansAutoInterp initialization."""
        autointerp = KMeansAutoInterp(
            cache_dir=temp_cache_dir,
            model=mock_model,
            centroids=dummy_centroids,
            layer_idx=5,
            activation_type="layer_output",
        )

        assert autointerp.num_centroids == 10
        assert autointerp.hidden_dim == 64
        assert autointerp.layer_idx == 5
        assert autointerp.cache.cache_dir == Path(temp_cache_dir)

    def test_assign_tokens_to_centroids(
        self, temp_cache_dir, mock_model, dummy_centroids
    ):
        """Test centroid assignment."""
        autointerp = KMeansAutoInterp(
            cache_dir=temp_cache_dir,
            model=mock_model,
            centroids=dummy_centroids,
        )

        # Create dummy activations
        seq_len = 5
        activations = th.randn(seq_len, 64)

        centroid_ids = autointerp.assign_tokens_to_centroids(activations)

        assert centroid_ids.shape == (seq_len,)
        assert centroid_ids.min() >= 0
        assert centroid_ids.max() < 10

    def test_get_top_k_validation_examples(
        self, temp_cache_dir, mock_model, dummy_centroids
    ):
        """Test getting top-k validation examples."""
        # Create validation data
        num_samples = 10
        seq_len = 8
        validation_activations = th.randn(num_samples, seq_len, 64)
        validation_tokens = th.randint(0, 1000, (num_samples, seq_len))

        autointerp = KMeansAutoInterp(
            cache_dir=temp_cache_dir,
            model=mock_model,
            centroids=dummy_centroids,
            validation_activations=validation_activations,
            validation_tokens=validation_tokens,
        )

        examples = autointerp.get_top_k_validation_examples(centroid_id=0, k=5)

        assert len(examples) == 5
        assert all(isinstance(ex, ValidationExample) for ex in examples)
        assert all(hasattr(ex, "text") for ex in examples)
        assert all(hasattr(ex, "activation") for ex in examples)

    @pytest.mark.asyncio
    async def test_generate_explanation_without_llm(
        self, temp_cache_dir, mock_model, dummy_centroids
    ):
        """Test explanation generation without LLM (heuristic mode)."""
        autointerp = KMeansAutoInterp(
            cache_dir=temp_cache_dir,
            model=mock_model,
            centroids=dummy_centroids,
            layer_idx=0,
        )

        # Create mock examples
        examples = [
            ValidationExample(
                text="test text",
                tokens=[1, 2, 3],
                token_strings=["test", "text", "."],
                activation=0.8,
                token_position=0,
            )
            for _ in range(5)
        ]

        explanation = await autointerp.generate_explanation(
            centroid_id=0, examples=examples, llm_client=None
        )

        assert explanation.centroid_id == 0
        assert isinstance(explanation.explanation, str)
        assert 0.0 <= explanation.confidence <= 1.0
        assert len(explanation.top_k_examples) > 0

    @pytest.mark.asyncio
    async def test_get_or_generate_explanation_caching(
        self, temp_cache_dir, mock_model, dummy_centroids
    ):
        """Test that explanations are properly cached."""
        num_samples = 10
        seq_len = 8
        validation_activations = th.randn(num_samples, seq_len, 64)
        validation_tokens = th.randint(0, 1000, (num_samples, seq_len))

        autointerp = KMeansAutoInterp(
            cache_dir=temp_cache_dir,
            model=mock_model,
            centroids=dummy_centroids,
            validation_activations=validation_activations,
            validation_tokens=validation_tokens,
        )

        # First call - should generate
        explanation1 = await autointerp.get_or_generate_explanation(centroid_id=0, k=5)

        # Second call - should use cache
        explanation2 = await autointerp.get_or_generate_explanation(centroid_id=0, k=5)

        assert explanation1.centroid_id == explanation2.centroid_id
        assert explanation1.explanation == explanation2.explanation

        # Force regenerate
        explanation3 = await autointerp.get_or_generate_explanation(
            centroid_id=0, k=5, force_regenerate=True
        )

        assert explanation3.centroid_id == 0
        # May or may not be the same explanation, but should be valid
        assert isinstance(explanation3.explanation, str)


def test_load_kmeans_centroids():
    """Test loading k-means centroids from file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name
        centroids = th.randn(50, 128)
        metadata = {"num_centroids": 50, "hidden_dim": 128, "layer": 5}

        th.save({"centroids": centroids, "metadata": metadata}, temp_path)

        try:
            loaded_centroids, loaded_metadata = load_kmeans_centroids(temp_path)

            assert loaded_centroids.shape == (50, 128)
            assert loaded_metadata["num_centroids"] == 50
            assert loaded_metadata["layer"] == 5
        finally:
            Path(temp_path).unlink()


def test_validation_example():
    """Test ValidationExample dataclass."""
    example = ValidationExample(
        text="Hello world",
        tokens=[1, 2],
        token_strings=["Hello", "world"],
        activation=0.95,
        token_position=1,
    )

    assert example.text == "Hello world"
    assert len(example.tokens) == 2
    assert example.activation == 0.95
    assert example.token_position == 1
