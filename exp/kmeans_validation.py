"""
Validation functions for k-means clustering.

These functions are used during k-means training to validate:
1. Loss behavior - detecting problematic monotonically increasing loss windows
2. Centroid quality - ensuring final centroids produce reasonable distributions
"""

from loguru import logger
import torch as th


def check_monotonic_increasing_window(
    losses: th.Tensor,
    window_size: int = 10,
) -> tuple[bool, int | None]:
    """
    Check if there is a window of consecutive iterations where loss is monotonically increasing.

    Args:
        losses: Tensor of shape (num_k_values, num_iterations) containing loss history
        window_size: Size of the window to check for monotonic increase

    Returns:
        Tuple of (has_problem, start_idx) where:
            - has_problem: True if a monotonically increasing window is found
            - start_idx: Starting index of the problematic window (None if no problem)
    """
    if losses.shape[1] < window_size:
        # Not enough data to check
        return False, None

    # Check each k value independently
    for k_idx in range(losses.shape[0]):
        k_losses = losses[k_idx]

        # Slide window through the loss history
        for start_idx in range(len(k_losses) - window_size + 1):
            window = k_losses[start_idx : start_idx + window_size]

            # Check if window is monotonically increasing
            is_monotonic_increasing = True
            for i in range(len(window) - 1):
                if window[i + 1] <= window[i]:
                    is_monotonic_increasing = False
                    break

            if is_monotonic_increasing:
                logger.warning(
                    f"K-means validation: Found monotonically increasing loss window "
                    f"for k_idx={k_idx}, starting at iteration {start_idx}, "
                    f"window_size={window_size}"
                )
                return True, start_idx

    return False, None


def validate_centroid_distribution(
    validation_data: th.Tensor,
    centroids: th.Tensor,
    min_assignment_ratio: float = 0.01,
    max_assignment_ratio: float = 0.5,
) -> tuple[bool, dict[str, float]]:
    """
    Validate that centroids produce a reasonable distribution of assignments on validation data.

    Args:
        validation_data: Tensor of shape (N, D) containing validation datapoints
        centroids: Tensor of shape (K, D) containing cluster centroids
        min_assignment_ratio: Minimum acceptable ratio of points assigned to any centroid
        max_assignment_ratio: Maximum acceptable ratio of points assigned to any centroid

    Returns:
        Tuple of (is_valid, stats) where:
            - is_valid: True if distribution is reasonable
            - stats: Dictionary containing distribution statistics
    """
    # Compute distances and assignments
    distances = th.cdist(validation_data.to(th.float32), centroids.to(th.float32), p=1)
    assignments = th.argmin(distances, dim=1)

    # Count assignments per centroid
    k = centroids.shape[0]
    assignment_counts = th.bincount(assignments, minlength=k)
    assignment_ratios = assignment_counts.float() / len(validation_data)

    # Check for empty centroids
    num_empty = (assignment_counts == 0).sum().item()

    # Check for over-concentrated centroids
    num_over_concentrated = (assignment_ratios > max_assignment_ratio).sum().item()

    # Check for under-utilized centroids (but not completely empty)
    num_under_utilized = (
        ((assignment_ratios > 0) & (assignment_ratios < min_assignment_ratio))
        .sum()
        .item()
    )

    # Compute statistics
    stats = {
        "num_empty_centroids": num_empty,
        "num_over_concentrated_centroids": num_over_concentrated,
        "num_under_utilized_centroids": num_under_utilized,
        "min_assignment_ratio": assignment_ratios[assignment_ratios > 0].min().item()
        if (assignment_ratios > 0).any()
        else 0.0,
        "max_assignment_ratio": assignment_ratios.max().item(),
        "mean_assignment_ratio": assignment_ratios.mean().item(),
        "std_assignment_ratio": assignment_ratios.std().item(),
    }

    is_valid = (
        num_empty == 0
        and num_over_concentrated == 0
        and num_under_utilized < k * 0.1  # Allow up to 10% under-utilized
    )

    if not is_valid:
        logger.warning(
            f"K-means validation: Centroid distribution is problematic. Stats: {stats}"
        )
    else:
        logger.info(
            f"K-means validation: Centroid distribution is reasonable. Stats: {stats}"
        )

    return is_valid, stats


def validate_kmeans_run(
    losses: th.Tensor,
    centroids: list[th.Tensor],
    validation_data: th.Tensor,
    window_size: int = 10,
    min_assignment_ratio: float = 0.01,
    max_assignment_ratio: float = 0.5,
) -> tuple[bool, dict]:
    """
    Run all validation checks on a k-means run.

    Args:
        losses: Tensor of shape (num_k_values, num_iterations) containing loss history
        centroids: List of tensors, each of shape (K, D) for different k values
        validation_data: Tensor of shape (N, D) containing validation datapoints
        window_size: Size of window to check for monotonic increase
        min_assignment_ratio: Minimum acceptable assignment ratio per centroid
        max_assignment_ratio: Maximum acceptable assignment ratio per centroid

    Returns:
        Tuple of (all_valid, validation_results) where:
            - all_valid: True if all validations pass
            - validation_results: Dictionary with detailed validation results
    """
    validation_results = {
        "loss_check": {},
        "distribution_checks": [],
    }

    # Check for monotonically increasing loss windows
    has_loss_problem, problem_start_idx = check_monotonic_increasing_window(
        losses, window_size
    )
    validation_results["loss_check"] = {
        "has_problem": has_loss_problem,
        "problem_start_idx": problem_start_idx,
    }

    # Check centroid distributions for each k value
    all_distributions_valid = True
    for k_idx, centroid_set in enumerate(centroids):
        is_valid, stats = validate_centroid_distribution(
            validation_data,
            centroid_set,
            min_assignment_ratio,
            max_assignment_ratio,
        )
        validation_results["distribution_checks"].append(
            {
                "k_idx": k_idx,
                "k_value": centroid_set.shape[0],
                "is_valid": is_valid,
                "stats": stats,
            }
        )
        all_distributions_valid = all_distributions_valid and is_valid

    all_valid = (not has_loss_problem) and all_distributions_valid

    if all_valid:
        logger.info("K-means validation: All checks passed ✓")
    else:
        logger.warning("K-means validation: Some checks failed ✗")

    return all_valid, validation_results
