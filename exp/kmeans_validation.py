"""
Validation functions for k-means clustering.

These functions are used during k-means training to validate:
1. Loss behavior - detecting problematic monotonically increasing loss windows
2. Centroid quality - ensuring final centroids produce reasonable distributions
"""

from dataclasses import dataclass

from loguru import logger
import torch as th
from tqdm import tqdm

WARNING_WINDOW_SIZE = 10
VALIDATION_SIZE_K_PROPORTION = 10


@dataclass
class CentroidValidationStats:
    """Statistics from centroid distribution validation."""

    num_empty_centroids: int
    num_over_concentrated_centroids: int
    num_under_utilized_centroids: int
    min_assignment_ratio: float
    max_assignment_ratio: float
    mean_assignment_ratio: float
    std_assignment_ratio: float
    entropy: float
    min_norm: float
    max_norm: float
    mean_norm: float
    std_norm: float

    def __str__(self) -> str:
        """Format stats for logging."""
        return (
            f"Empty: {self.num_empty_centroids}, "
            f"Over-concentrated: {self.num_over_concentrated_centroids}, "
            f"Under-utilized: {self.num_under_utilized_centroids}, "
            f"Min ratio: {self.min_assignment_ratio:.4f}, "
            f"Max ratio: {self.max_assignment_ratio:.4f}, "
            f"Mean ratio: {self.mean_assignment_ratio:.4f}, "
            f"Std ratio: {self.std_assignment_ratio:.4f}, "
            f"Entropy: {self.entropy:.4f}, "
            f"Min norm: {self.min_norm:.4f}, "
            f"Max norm: {self.max_norm:.4f}, "
            f"Mean norm: {self.mean_norm:.4f}, "
            f"Std norm: {self.std_norm:.4f}"
        )


def check_monotonic_increasing_window(
    losses: th.Tensor,
    window_size: int = WARNING_WINDOW_SIZE,
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
    if th.isnan(losses).any():
        logger.warning("K-means validation: Found NaN in losses")
        return False, None

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
    min_assignment_ratio: float = 0.05,
    max_assignment_ratio: float = 20.0,
    minibatch_size: int = 100000,
    centroid_minibatch_size: int = 16384,
) -> tuple[bool, CentroidValidationStats]:
    """
    Validate that centroids produce a reasonable distribution of assignments on validation data.

    Args:
        validation_data: Tensor of shape (N, D) containing validation datapoints
        centroids: Tensor of shape (K, D) containing cluster centroids
        min_assignment_ratio: Minimum acceptable ratio (defaults to 0.05)
        max_assignment_ratio: Maximum acceptable ratio (defaults to 20.0)
        minibatch_size: Size of minibatches for GPU processing (defaults to 100000)
        centroid_minibatch_size: Size of centroid chunks to avoid CUDA limits (defaults to 16384)

    Returns:
        Tuple of (is_valid, stats) where:
            - is_valid: True if distribution is reasonable
            - stats: CentroidValidationStats containing distribution statistics
    """
    # check for any nans
    if th.isnan(validation_data).any() or th.isnan(centroids).any():
        logger.warning("K-means validation: Found NaN in validation data or centroids")
        return False, CentroidValidationStats(
            num_empty_centroids=0,
            num_over_concentrated_centroids=0,
            num_under_utilized_centroids=0,
            min_assignment_ratio=0.0,
            max_assignment_ratio=0.0,
            mean_assignment_ratio=0.0,
            std_assignment_ratio=0.0,
            entropy=0.0,
            min_norm=0.0,
            max_norm=0.0,
            mean_norm=0.0,
            std_norm=0.0,
        )

    # Use GPU if available, otherwise CPU
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Move centroids to device
    centroids_gpu = centroids.to(device).to(th.float32)

    # Process validation data in minibatches to avoid OOM
    all_assignments = []
    n_samples = validation_data.shape[0]

    logger.debug(
        f"🚀 GPU validation: Processing {n_samples} samples in batches of {minibatch_size}, centroids in chunks of {centroid_minibatch_size}"
    )

    for start_idx in tqdm(
        range(0, n_samples, minibatch_size), desc="Validation batches", leave=False
    ):
        end_idx = min(start_idx + minibatch_size, n_samples)
        batch_data = validation_data[start_idx:end_idx].to(device).to(th.float32)

        # Compute distances for this batch, chunking centroids to avoid CUDA limits
        if centroids_gpu.shape[0] <= centroid_minibatch_size:
            # Small enough to compute in one go
            batch_distances = th.cdist(batch_data, centroids_gpu, p=1)
        else:
            # Chunk centroids to avoid CUDA configuration limits
            all_distances = []
            n_centroids = centroids_gpu.shape[0]

            for c_start in range(0, n_centroids, centroid_minibatch_size):
                c_end = min(c_start + centroid_minibatch_size, n_centroids)
                centroid_chunk = centroids_gpu[c_start:c_end]
                chunk_distances = th.cdist(batch_data, centroid_chunk, p=1)
                all_distances.append(chunk_distances)

            # Concatenate along centroid dimension
            batch_distances = th.cat(all_distances, dim=1)

        batch_assignments = th.argmin(batch_distances, dim=1)

        # Move back to CPU to save GPU memory
        all_assignments.append(batch_assignments.cpu())

        # Clear GPU cache
        del batch_data, batch_distances, batch_assignments
        if device.type == "cuda":
            th.cuda.empty_cache()

    # Concatenate all assignments
    assignments = th.cat(all_assignments, dim=0)

    logger.debug("✅ GPU validation completed successfully")

    # Count assignments per centroid
    k = centroids.shape[0]
    assignment_counts = th.bincount(assignments, minlength=k)
    assignment_ratios = assignment_counts.float() / len(validation_data)

    # Always divide by k to get the actual thresholds
    min_assignment_threshold = min_assignment_ratio / k
    max_assignment_threshold = max_assignment_ratio / k

    # Check for empty centroids
    num_empty = (assignment_counts == 0).sum().item()

    # Check for over-concentrated centroids
    num_over_concentrated = (assignment_ratios > max_assignment_threshold).sum().item()

    # Check for under-utilized centroids (but not completely empty)
    num_under_utilized = (
        ((assignment_ratios > 0) & (assignment_ratios < min_assignment_threshold))
        .sum()
        .item()
    )

    # Calculate entropy of assignment distribution
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    probs = assignment_ratios + epsilon
    entropy = -(probs * th.log(probs)).sum().item()

    # Calculate norm of assignment distribution
    norms = th.norm(centroids, p=2, dim=1)
    min_norm = norms.min().item()
    max_norm = norms.max().item()
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()

    # Create statistics dataclass
    stats = CentroidValidationStats(
        num_empty_centroids=num_empty,
        num_over_concentrated_centroids=num_over_concentrated,
        num_under_utilized_centroids=num_under_utilized,
        min_assignment_ratio=assignment_ratios[assignment_ratios > 0].min().item()
        if (assignment_ratios > 0).any()
        else 0.0,
        max_assignment_ratio=assignment_ratios.max().item(),
        mean_assignment_ratio=assignment_ratios.mean().item(),
        std_assignment_ratio=assignment_ratios.std().item(),
        entropy=entropy,
        min_norm=min_norm,
        max_norm=max_norm,
        mean_norm=mean_norm,
        std_norm=std_norm,
    )

    is_valid = (
        num_empty == 0
        and num_over_concentrated == 0
        and num_under_utilized < k * 0.1  # Allow up to 10% under-utilized
    )

    if not is_valid:
        logger.warning(
            f"K-means validation: Centroid distribution is problematic. {stats}"
        )
    else:
        logger.info(f"K-means validation: Centroid distribution is reasonable. {stats}")

    return is_valid, stats
