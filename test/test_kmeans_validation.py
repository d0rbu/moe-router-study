"""
Unit tests for the k-means validation framework.

Tests cover loss monotonicity validation, centroid distribution validation,
and integration with the k-means process.
"""

import pytest
import torch as th

from exp.kmeans_validation import (
    CentroidDistributionValidator,
    KMeansValidator,
    LossMonotonicityValidator,
    ValidationConfig,
)


class TestValidationConfig:
    """Test ValidationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()

        assert config.enabled is True
        assert config.validation_frequency == 1
        assert config.monotonicity_window_size == 10
        assert config.monotonicity_threshold == 0.01
        assert config.validation_set_ratio == 0.1
        assert config.min_points_per_centroid == 5
        assert config.max_empty_centroids_ratio == 0.1
        assert config.distribution_balance_threshold == 0.05

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            enabled=False,
            validation_frequency=5,
            monotonicity_window_size=20,
            monotonicity_threshold=0.05,
        )

        assert config.enabled is False
        assert config.validation_frequency == 5
        assert config.monotonicity_window_size == 20
        assert config.monotonicity_threshold == 0.05


class TestLossMonotonicityValidator:
    """Test LossMonotonicityValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        config = ValidationConfig()
        validator = LossMonotonicityValidator(config)

        assert validator.config == config
        assert len(validator.loss_history) == 0

    def test_update_losses(self):
        """Test loss history updates."""
        config = ValidationConfig(monotonicity_window_size=3)
        validator = LossMonotonicityValidator(config)

        # Add some losses
        losses1 = th.tensor([1.0, 2.0])
        losses2 = th.tensor([0.9, 1.8])
        losses3 = th.tensor([0.8, 1.6])
        losses4 = th.tensor([0.7, 1.4])

        validator.update_losses(losses1)
        validator.update_losses(losses2)
        validator.update_losses(losses3)
        validator.update_losses(losses4)

        # Should keep only window_size + 1 = 4 losses
        assert len(validator.loss_history) == 4

        # Add one more to test trimming
        losses5 = th.tensor([0.6, 1.2])
        validator.update_losses(losses5)

        # Should still be 4, with oldest removed
        assert len(validator.loss_history) == 4
        assert th.allclose(validator.loss_history[0], losses2)
        assert th.allclose(validator.loss_history[-1], losses5)

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data."""
        config = ValidationConfig(monotonicity_window_size=5)
        validator = LossMonotonicityValidator(config)

        # Add only 3 losses (less than window size)
        validator.update_losses(th.tensor([1.0, 2.0]))
        validator.update_losses(th.tensor([0.9, 1.8]))
        validator.update_losses(th.tensor([0.8, 1.6]))

        result = validator.validate()
        assert result["status"] == "insufficient_data"
        assert result["warnings"] == []

    def test_validate_no_monotonic_increase(self):
        """Test validation with no monotonic increases."""
        config = ValidationConfig(monotonicity_window_size=3)
        validator = LossMonotonicityValidator(config)

        # Add decreasing losses (good behavior)
        validator.update_losses(th.tensor([1.0, 2.0]))
        validator.update_losses(th.tensor([0.9, 1.8]))
        validator.update_losses(th.tensor([0.8, 1.6]))

        result = validator.validate()
        assert result["status"] == "validated"
        assert len(result["warnings"]) == 0
        assert "k_0_recent_trend" in result["metrics"]
        assert "k_1_recent_trend" in result["metrics"]

    def test_validate_monotonic_increase_below_threshold(self):
        """Test validation with monotonic increase below threshold."""
        config = ValidationConfig(
            monotonicity_window_size=3, monotonicity_threshold=0.1
        )
        validator = LossMonotonicityValidator(config)

        # Add slightly increasing losses (below threshold)
        validator.update_losses(th.tensor([1.0, 2.0]))
        validator.update_losses(th.tensor([1.005, 2.005]))  # 0.5% increase
        validator.update_losses(th.tensor([1.01, 2.01]))  # 1% total increase

        result = validator.validate()
        assert result["status"] == "validated"
        assert len(result["warnings"]) == 0  # Below threshold

    def test_validate_monotonic_increase_above_threshold(self):
        """Test validation with monotonic increase above threshold."""
        config = ValidationConfig(
            monotonicity_window_size=3, monotonicity_threshold=0.01
        )
        validator = LossMonotonicityValidator(config)

        # Add significantly increasing losses (above threshold)
        validator.update_losses(th.tensor([1.0, 2.0]))
        validator.update_losses(th.tensor([1.1, 2.1]))  # 10% increase
        validator.update_losses(th.tensor([1.2, 2.2]))  # 20% total increase

        result = validator.validate()
        assert result["status"] == "validated"
        assert len(result["warnings"]) == 2  # One for each k value

        # Check warning messages
        for warning in result["warnings"]:
            assert "Loss increased monotonically" in warning
            assert "relative increase:" in warning

    def test_check_monotonic_increases(self):
        """Test monotonic increase detection."""
        config = ValidationConfig()
        validator = LossMonotonicityValidator(config)

        # Test monotonic increase
        assert validator._check_monotonic_increases([1.0, 1.1, 1.2, 1.3]) is True

        # Test non-monotonic (decrease)
        assert validator._check_monotonic_increases([1.0, 1.1, 1.0, 1.3]) is False

        # Test non-monotonic (plateau)
        assert validator._check_monotonic_increases([1.0, 1.1, 1.1, 1.3]) is False

        # Test single value
        assert validator._check_monotonic_increases([1.0]) is False

        # Test empty
        assert validator._check_monotonic_increases([]) is False

    def test_compute_trend(self):
        """Test trend computation."""
        config = ValidationConfig()
        validator = LossMonotonicityValidator(config)

        # Test increasing trend
        trend = validator._compute_trend([1.0, 2.0, 3.0, 4.0])
        assert trend > 0

        # Test decreasing trend
        trend = validator._compute_trend([4.0, 3.0, 2.0, 1.0])
        assert trend < 0

        # Test flat trend
        trend = validator._compute_trend([2.0, 2.0, 2.0, 2.0])
        assert abs(trend) < 1e-6

        # Test single value
        trend = validator._compute_trend([1.0])
        assert trend == 0.0


class TestCentroidDistributionValidator:
    """Test CentroidDistributionValidator."""

    def test_initialization(self):
        """Test validator initialization."""
        config = ValidationConfig()
        validator = CentroidDistributionValidator(config)

        assert validator.config == config
        assert validator.validation_data is None

    def test_set_validation_data(self):
        """Test setting validation data."""
        config = ValidationConfig(validation_set_ratio=0.2)
        validator = CentroidDistributionValidator(config)

        # Create test data
        data = th.randn(1000, 10)
        validator.set_validation_data(data)

        assert validator.validation_data is not None
        expected_size = int(1000 * 0.2)
        assert len(validator.validation_data) == expected_size
        assert validator.validation_data.shape[1] == 10

    def test_set_validation_data_small_dataset(self):
        """Test setting validation data with small dataset."""
        config = ValidationConfig(validation_set_ratio=0.1)
        validator = CentroidDistributionValidator(config)

        # Create very small dataset
        data = th.randn(5, 10)
        validator.set_validation_data(data)

        assert validator.validation_data is not None
        # Should use minimum of 1000 or dataset size
        assert len(validator.validation_data) == 5

    @pytest.mark.asyncio
    async def test_validate_no_validation_data(self):
        """Test validation without validation data."""
        config = ValidationConfig()
        validator = CentroidDistributionValidator(config)

        centroids_list = [th.randn(5, 10)]
        result = await validator.validate(centroids_list)

        assert result["status"] == "no_validation_data"
        assert result["warnings"] == []

    @pytest.mark.asyncio
    async def test_validate_good_distribution(self):
        """Test validation with good centroid distribution."""
        config = ValidationConfig(
            validation_set_ratio=1.0,  # Use all data for validation
            min_points_per_centroid=1,
            max_empty_centroids_ratio=0.0,
        )
        validator = CentroidDistributionValidator(config)

        # Create well-separated centroids and data
        centroids = th.tensor(
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [0.0, 5.0],
            ],
            dtype=th.float32,
        )

        # Create data points near each centroid
        data = th.cat(
            [
                th.randn(50, 2) + th.tensor([0.0, 0.0]),  # Near first centroid
                th.randn(50, 2) + th.tensor([5.0, 0.0]),  # Near second centroid
                th.randn(50, 2) + th.tensor([0.0, 5.0]),  # Near third centroid
            ]
        )

        validator.set_validation_data(data)
        result = await validator.validate([centroids])

        assert result["status"] == "validated"
        assert len(result["warnings"]) == 0
        assert "k_0_empty_centroids" in result["metrics"]
        assert result["metrics"]["k_0_empty_centroids"] == 0

    @pytest.mark.asyncio
    async def test_validate_empty_centroids(self):
        """Test validation with empty centroids."""
        config = ValidationConfig(
            validation_set_ratio=1.0,
            max_empty_centroids_ratio=0.1,  # Allow 10% empty
        )
        validator = CentroidDistributionValidator(config)

        # Create centroids where some are far from data
        centroids = th.tensor(
            [
                [0.0, 0.0],
                [100.0, 100.0],  # Far from data
                [200.0, 200.0],  # Far from data
            ],
            dtype=th.float32,
        )

        # Create data only near first centroid
        data = th.randn(100, 2) + th.tensor([0.0, 0.0])

        validator.set_validation_data(data)
        result = await validator.validate([centroids])

        assert result["status"] == "validated"
        # Should have warnings about empty centroids (2/3 = 66% > 10%)
        assert len(result["warnings"]) > 0
        assert any(
            "have no assigned validation points" in w for w in result["warnings"]
        )

    @pytest.mark.asyncio
    async def test_validate_imbalanced_distribution(self):
        """Test validation with imbalanced centroid assignments."""
        config = ValidationConfig(
            validation_set_ratio=1.0,
            distribution_balance_threshold=0.01,  # Very low threshold
        )
        validator = CentroidDistributionValidator(config)

        # Create centroids
        centroids = th.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=th.float32,
        )

        # Create highly imbalanced data (most points near first centroid)
        data = th.cat(
            [
                th.randn(95, 2) + th.tensor([0.0, 0.0]),  # 95% near first
                th.randn(5, 2) + th.tensor([1.0, 0.0]),  # 5% near second
            ]
        )

        validator.set_validation_data(data)
        result = await validator.validate([centroids])

        assert result["status"] == "validated"
        # Should have warnings about imbalanced assignments
        assert len(result["warnings"]) > 0
        assert any(
            "Highly imbalanced centroid assignments" in w for w in result["warnings"]
        )


class TestKMeansValidator:
    """Test KMeansValidator integration."""

    def test_initialization(self):
        """Test validator initialization."""
        config = ValidationConfig()
        validator = KMeansValidator(config)

        assert validator.config == config
        assert validator.iteration_count == 0
        assert isinstance(validator.loss_validator, LossMonotonicityValidator)
        assert isinstance(
            validator.distribution_validator, CentroidDistributionValidator
        )

    def test_set_validation_data(self):
        """Test setting validation data."""
        config = ValidationConfig()
        validator = KMeansValidator(config)

        data = th.randn(100, 10)
        validator.set_validation_data(data)

        assert validator.distribution_validator.validation_data is not None

    def test_set_validation_data_disabled(self):
        """Test setting validation data when disabled."""
        config = ValidationConfig(enabled=False)
        validator = KMeansValidator(config)

        data = th.randn(100, 10)
        validator.set_validation_data(data)

        # Should not set validation data when disabled
        assert validator.distribution_validator.validation_data is None

    def test_should_validate(self):
        """Test validation frequency logic."""
        config = ValidationConfig(validation_frequency=3)
        validator = KMeansValidator(config)

        # Initially should validate (iteration 0)
        assert validator.should_validate() is True

        # Increment iteration count manually to test frequency
        validator.iteration_count = 1
        assert validator.should_validate() is False

        validator.iteration_count = 2
        assert validator.should_validate() is False

        validator.iteration_count = 3
        assert validator.should_validate() is True

        validator.iteration_count = 6
        assert validator.should_validate() is True

    def test_should_validate_disabled(self):
        """Test validation when disabled."""
        config = ValidationConfig(enabled=False)
        validator = KMeansValidator(config)

        assert validator.should_validate() is False

    @pytest.mark.asyncio
    async def test_validate_iteration_skipped(self):
        """Test validation iteration when skipped."""
        config = ValidationConfig(validation_frequency=2)
        validator = KMeansValidator(config)

        losses = th.tensor([1.0, 2.0])
        centroids_list = [th.randn(3, 10), th.randn(5, 10)]

        # First call should be skipped (iteration 1 % 2 != 0)
        result = await validator.validate_iteration(losses, centroids_list)
        assert result["status"] == "skipped"

        # Second call should validate (iteration 2 % 2 == 0)
        result = await validator.validate_iteration(losses, centroids_list)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_validate_iteration_complete(self):
        """Test complete validation iteration."""
        config = ValidationConfig(validation_frequency=1)
        validator = KMeansValidator(config)

        # Set up validation data
        data = th.randn(100, 10)
        validator.set_validation_data(data)

        losses = th.tensor([1.0, 2.0])
        centroids_list = [th.randn(3, 10), th.randn(5, 10)]

        result = await validator.validate_iteration(losses, centroids_list)

        assert result["status"] == "completed"
        assert result["iteration"] == 1
        assert "warnings" in result
        assert "metrics" in result
        assert "loss_validation" in result
        assert "distribution_validation" in result

    def test_get_summary(self):
        """Test validation summary."""
        config = ValidationConfig(
            enabled=True,
            validation_frequency=5,
            monotonicity_window_size=15,
            validation_set_ratio=0.2,
        )
        validator = KMeansValidator(config)

        # Set some validation data
        data = th.randn(100, 10)
        validator.set_validation_data(data)

        # Add some loss history
        validator.loss_validator.update_losses(th.tensor([1.0, 2.0]))
        validator.iteration_count = 3

        summary = validator.get_summary()

        assert summary["config"]["enabled"] is True
        assert summary["config"]["validation_frequency"] == 5
        assert summary["config"]["monotonicity_window_size"] == 15
        assert summary["config"]["validation_set_ratio"] == 0.2

        assert summary["state"]["iteration_count"] == 3
        assert summary["state"]["loss_history_length"] == 1
        assert summary["state"]["has_validation_data"] is True


class TestIntegration:
    """Integration tests for the validation framework."""

    @pytest.mark.asyncio
    async def test_full_validation_cycle(self):
        """Test a full validation cycle with realistic data."""
        config = ValidationConfig(
            enabled=True,
            validation_frequency=1,
            monotonicity_window_size=5,
            monotonicity_threshold=0.05,
            validation_set_ratio=0.1,
        )
        validator = KMeansValidator(config)

        # Set up validation data
        data = th.randn(1000, 20)
        validator.set_validation_data(data)

        # Simulate several iterations of k-means
        results = []
        for i in range(10):
            # Simulate decreasing losses (good behavior)
            losses = th.tensor([10.0 - i * 0.5, 20.0 - i * 1.0])

            # Create some centroids
            centroids_list = [
                th.randn(5, 20),  # 5 centroids for first k value
                th.randn(10, 20),  # 10 centroids for second k value
            ]

            result = await validator.validate_iteration(losses, centroids_list)
            results.append(result)

        # Check that all iterations completed successfully
        completed_results = [r for r in results if r["status"] == "completed"]
        assert len(completed_results) == 10

        # Should have no warnings for decreasing losses
        all_warnings = []
        for result in completed_results:
            all_warnings.extend(result.get("warnings", []))

        # Filter out distribution warnings (which are expected with random data)
        loss_warnings = [w for w in all_warnings if "Loss increased monotonically" in w]
        assert len(loss_warnings) == 0

    @pytest.mark.asyncio
    async def test_validation_with_problematic_losses(self):
        """Test validation with problematic loss patterns."""
        config = ValidationConfig(
            enabled=True,
            validation_frequency=1,
            monotonicity_window_size=3,
            monotonicity_threshold=0.01,
        )
        validator = KMeansValidator(config)

        # Set up validation data
        data = th.randn(100, 10)
        validator.set_validation_data(data)

        # Simulate monotonically increasing losses (bad behavior)
        base_losses = [1.0, 2.0]
        results = []

        for i in range(5):
            # Increasing losses
            losses = th.tensor([base_losses[0] * (1.1**i), base_losses[1] * (1.1**i)])
            centroids_list = [th.randn(3, 10), th.randn(5, 10)]

            result = await validator.validate_iteration(losses, centroids_list)
            results.append(result)

        # Should detect monotonic increases after window is full
        later_results = results[3:]  # After window is full

        warnings_found = False
        for result in later_results:
            if result["status"] == "completed" and result.get("warnings"):
                loss_warnings = [
                    w for w in result["warnings"] if "Loss increased monotonically" in w
                ]
                if loss_warnings:
                    warnings_found = True
                    break

        assert warnings_found, "Should have detected monotonic loss increases"


if __name__ == "__main__":
    pytest.main([__file__])
