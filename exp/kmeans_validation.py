"""
K-means validation framework for monitoring training quality and detecting issues.

This module provides validation capabilities that run during k-means training to:
1. Detect problematic loss patterns (monotonic increases over windows)
2. Validate centroid distribution quality using validation data
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger
import torch as th


@dataclass
class ValidationConfig:
    """Configuration for k-means validation."""

    # General validation settings
    enabled: bool = True
    validation_frequency: int = 1  # Validate every N iterations

    # Loss monotonicity validation
    monotonicity_window_size: int = 10  # Window size to check for monotonic increases
    monotonicity_threshold: float = (
        0.01  # Minimum relative increase to consider significant
    )

    # Centroid distribution validation
    validation_set_ratio: float = 0.1  # Fraction of data to use for validation
    min_points_per_centroid: int = (
        5  # Minimum points that should be assigned to each centroid
    )
    max_empty_centroids_ratio: float = (
        0.1  # Maximum fraction of centroids that can be empty
    )
    distribution_balance_threshold: float = (
        0.05  # Threshold for detecting severely imbalanced assignments
    )


class LossMonotonicityValidator:
    """Validates that losses don't increase monotonically over extended windows."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.loss_history: list[th.Tensor] = []

    def update_losses(self, losses: th.Tensor) -> None:
        """Update the loss history with new losses."""
        self.loss_history.append(losses.clone())

        # Keep only the window we need for validation
        max_history = self.config.monotonicity_window_size + 1
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]

    def validate(self) -> dict[str, Any]:
        """
        Validate loss monotonicity patterns.

        Returns:
            Dictionary with validation results including warnings and metrics.
        """
        if len(self.loss_history) < self.config.monotonicity_window_size:
            return {"status": "insufficient_data", "warnings": []}

        warnings_list = []
        metrics = {}

        # Check each k value separately
        num_k_values = self.loss_history[0].shape[0]

        for k_idx in range(num_k_values):
            k_losses = [losses[k_idx].item() for losses in self.loss_history]

            # Check for monotonic increases in the window
            monotonic_increases = self._check_monotonic_increases(k_losses)

            if monotonic_increases:
                relative_increase = (k_losses[-1] - k_losses[0]) / abs(k_losses[0])

                if relative_increase > self.config.monotonicity_threshold:
                    warning_msg = (
                        f"K-value {k_idx}: Loss increased monotonically over "
                        f"{self.config.monotonicity_window_size} iterations "
                        f"(relative increase: {relative_increase:.4f})"
                    )
                    warnings_list.append(warning_msg)
                    logger.warning(warning_msg)

            # Store metrics for this k value
            metrics[f"k_{k_idx}_recent_trend"] = self._compute_trend(k_losses)
            metrics[f"k_{k_idx}_relative_change"] = (
                (k_losses[-1] - k_losses[0]) / abs(k_losses[0])
                if len(k_losses) >= 2 and k_losses[0] != 0
                else 0.0
            )

        return {"status": "validated", "warnings": warnings_list, "metrics": metrics}

    def _check_monotonic_increases(self, losses: list[float]) -> bool:
        """Check if losses are monotonically increasing."""
        if len(losses) < 2:
            return False

        return all(losses[i] > losses[i - 1] for i in range(1, len(losses)))

    def _compute_trend(self, losses: list[float]) -> float:
        """Compute the trend (slope) of recent losses."""
        if len(losses) < 2:
            return 0.0

        # Simple linear trend computation
        n = len(losses)
        x_mean = (n - 1) / 2
        y_mean = sum(losses) / n

        numerator = sum((i - x_mean) * (losses[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0


class CentroidDistributionValidator:
    """Validates that centroids are reasonably distributed using validation data."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_data: th.Tensor | None = None

    def set_validation_data(self, data: th.Tensor) -> None:
        """Set the validation dataset."""
        # Sample validation data based on the configured ratio
        num_validation_samples = int(len(data) * self.config.validation_set_ratio)
        if num_validation_samples == 0:
            num_validation_samples = min(1000, len(data))  # Minimum validation set

        # Random sampling for validation
        indices = th.randperm(len(data))[:num_validation_samples]
        self.validation_data = data[indices].clone()

        logger.debug(f"Set validation data with {len(self.validation_data)} samples")

    async def validate(self, centroids_list: list[th.Tensor]) -> dict[str, Any]:
        """
        Validate centroid distribution quality.

        Args:
            centroids_list: List of centroid tensors for different k values

        Returns:
            Dictionary with validation results including warnings and metrics.
        """
        if self.validation_data is None:
            return {"status": "no_validation_data", "warnings": []}

        warnings_list = []
        metrics = {}

        for k_idx, centroids in enumerate(centroids_list):
            validation_result = await self._validate_single_k(centroids, k_idx)

            warnings_list.extend(validation_result["warnings"])
            metrics.update(validation_result["metrics"])

        return {"status": "validated", "warnings": warnings_list, "metrics": metrics}

    async def _validate_single_k(
        self, centroids: th.Tensor, k_idx: int
    ) -> dict[str, Any]:
        """Validate distribution for a single k value."""
        warnings_list = []
        metrics = {}

        # Compute assignments for validation data
        distances = th.cdist(
            self.validation_data.to(th.float32), centroids.to(th.float32), p=1
        )
        assignments = th.argmin(distances, dim=1)

        # Count assignments per centroid
        num_centroids = centroids.shape[0]
        assignment_counts = th.bincount(assignments, minlength=num_centroids)

        # Check for empty centroids
        empty_centroids = (assignment_counts == 0).sum().item()
        empty_ratio = empty_centroids / num_centroids

        if empty_ratio > self.config.max_empty_centroids_ratio:
            warning_msg = (
                f"K-value {k_idx}: {empty_centroids}/{num_centroids} centroids "
                f"({empty_ratio:.2%}) have no assigned validation points"
            )
            warnings_list.append(warning_msg)
            logger.warning(warning_msg)

        # Check for severely under-populated centroids
        under_populated = (
            (assignment_counts < self.config.min_points_per_centroid).sum().item()
        )
        under_populated_ratio = under_populated / num_centroids

        if under_populated_ratio > self.config.max_empty_centroids_ratio:
            warning_msg = (
                f"K-value {k_idx}: {under_populated}/{num_centroids} centroids "
                f"({under_populated_ratio:.2%}) have fewer than {self.config.min_points_per_centroid} "
                f"assigned validation points"
            )
            warnings_list.append(warning_msg)
            logger.warning(warning_msg)

        # Check distribution balance
        if assignment_counts.numel() > 0:
            mean_assignments = assignment_counts.float().mean()
            std_assignments = assignment_counts.float().std()
            cv = (
                std_assignments / mean_assignments
                if mean_assignments > 0
                else float("inf")
            )

            if cv > self.config.distribution_balance_threshold:
                warning_msg = (
                    f"K-value {k_idx}: Highly imbalanced centroid assignments "
                    f"(coefficient of variation: {cv:.4f})"
                )
                warnings_list.append(warning_msg)
                logger.warning(warning_msg)

            # Store metrics
            metrics[f"k_{k_idx}_empty_centroids"] = empty_centroids
            metrics[f"k_{k_idx}_empty_ratio"] = empty_ratio
            metrics[f"k_{k_idx}_under_populated"] = under_populated
            metrics[f"k_{k_idx}_assignment_cv"] = cv
            metrics[f"k_{k_idx}_min_assignments"] = assignment_counts.min().item()
            metrics[f"k_{k_idx}_max_assignments"] = assignment_counts.max().item()
            metrics[f"k_{k_idx}_mean_assignments"] = mean_assignments.item()

        return {"warnings": warnings_list, "metrics": metrics}


class KMeansValidator:
    """Main validator that coordinates all validation checks."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.loss_validator = LossMonotonicityValidator(config)
        self.distribution_validator = CentroidDistributionValidator(config)
        self.iteration_count = 0

    def set_validation_data(self, data: th.Tensor) -> None:
        """Set validation data for distribution validation."""
        if self.config.enabled:
            self.distribution_validator.set_validation_data(data)

    def should_validate(self) -> bool:
        """Check if validation should run on this iteration."""
        return (
            self.config.enabled
            and self.iteration_count % self.config.validation_frequency == 0
        )

    async def validate_iteration(
        self, losses: th.Tensor, centroids_list: list[th.Tensor]
    ) -> dict[str, Any]:
        """
        Run validation for the current iteration.

        Args:
            losses: Current losses for all k values
            centroids_list: List of centroids for all k values

        Returns:
            Combined validation results from all validators.
        """
        self.iteration_count += 1

        if not self.should_validate():
            return {"status": "skipped"}

        # Update loss history
        self.loss_validator.update_losses(losses)

        # Run validations
        loss_results = self.loss_validator.validate()
        distribution_results = await self.distribution_validator.validate(
            centroids_list
        )

        # Combine results
        all_warnings = loss_results.get("warnings", []) + distribution_results.get(
            "warnings", []
        )
        all_metrics = {
            **loss_results.get("metrics", {}),
            **distribution_results.get("metrics", {}),
        }

        # Log summary if there are warnings
        if all_warnings:
            logger.warning(
                f"Validation iteration {self.iteration_count}: {len(all_warnings)} warnings detected"
            )
            for warning in all_warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.debug(
                f"Validation iteration {self.iteration_count}: No issues detected"
            )

        return {
            "status": "completed",
            "iteration": self.iteration_count,
            "warnings": all_warnings,
            "metrics": all_metrics,
            "loss_validation": loss_results,
            "distribution_validation": distribution_results,
        }

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of validation state."""
        return {
            "config": {
                "enabled": self.config.enabled,
                "validation_frequency": self.config.validation_frequency,
                "monotonicity_window_size": self.config.monotonicity_window_size,
                "validation_set_ratio": self.config.validation_set_ratio,
            },
            "state": {
                "iteration_count": self.iteration_count,
                "loss_history_length": len(self.loss_validator.loss_history),
                "has_validation_data": self.distribution_validator.validation_data
                is not None,
            },
        }
