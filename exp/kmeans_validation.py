"""
Validation functions for k-means clustering.

These functions are used during k-means training to validate:
1. Loss behavior - detecting problematic monotonically increasing loss windows
2. Centroid quality - ensuring final centroids produce reasonable distributions
"""

from dataclasses import dataclass
from typing import List, Tuple
import math

from loguru import logger
import torch as th
from tqdm import tqdm

from core.device import DeviceType, get_backend, get_device

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


def _compute_batch_centroid_distances(
    batch_data: th.Tensor,
    centroid_chunk: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """
    Compute distances between a batch of data points and a chunk of centroids.
    
    Args:
        batch_data: Tensor of shape (batch_size, D) on CPU
        centroid_chunk: Tensor of shape (chunk_size, D) on CPU  
        device: Device to perform computation on
        
    Returns:
        Distance tensor of shape (batch_size, chunk_size) on CPU
    """
    # Move data to device
    batch_gpu = batch_data.to(device).to(th.float32)
    centroids_gpu = centroid_chunk.to(device).to(th.float32)
    
    # Compute distances (Manhattan/L1 distance)
    distances = th.cdist(batch_gpu, centroids_gpu, p=1)
    
    # Move result back to CPU
    return distances.cpu()


def _distribute_work_across_devices(
    validation_data: th.Tensor,
    centroids: th.Tensor,
    minibatch_size: int,
    centroid_minibatch_size: int,
    device_type: DeviceType,
) -> List[th.Tensor]:
    """
    Distribute centroid validation work across available devices.
    
    This function explicitly splits work along two dimensions:
    1. Batch dimension: Split validation data into minibatches
    2. Centroid dimension: Split centroids into chunks
    
    Each (batch, centroid_chunk) pair is processed on an available device.
    
    Args:
        validation_data: Tensor of shape (N, D) containing validation datapoints
        centroids: Tensor of shape (K, D) containing cluster centroids
        minibatch_size: Size of data batches
        centroid_minibatch_size: Size of centroid chunks
        device_type: Device type to use ("cuda" or "xpu")
        
    Returns:
        List of assignment tensors, one per data batch
    """
    backend = get_backend(device_type)
    
    # Get available devices
    if backend.is_available():
        num_devices = backend.device_count()
        devices = [get_device(device_type, i) for i in range(num_devices)]
        logger.debug(f"Using {num_devices} {device_type.upper()} devices for validation")
    else:
        devices = [th.device("cpu")]
        logger.debug("Using CPU for validation (no GPU devices available)")
    
    n_samples = validation_data.shape[0]
    n_centroids = centroids.shape[0]
    
    # Split data into batches
    n_batches = math.ceil(n_samples / minibatch_size)
    data_batches = [
        validation_data[i * minibatch_size:(i + 1) * minibatch_size]
        for i in range(n_batches)
    ]
    
    # Split centroids into chunks
    n_centroid_chunks = math.ceil(n_centroids / centroid_minibatch_size)
    centroid_chunks = [
        centroids[i * centroid_minibatch_size:(i + 1) * centroid_minibatch_size]
        for i in range(n_centroid_chunks)
    ]
    
    logger.debug(
        f"Split work: {n_batches} data batches × {n_centroid_chunks} centroid chunks "
        f"= {n_batches * n_centroid_chunks} total tasks across {len(devices)} devices"
    )
    
    # Assertions for correctness
    assert len(data_batches) == n_batches, f"Expected {n_batches} data batches, got {len(data_batches)}"
    assert len(centroid_chunks) == n_centroid_chunks, f"Expected {n_centroid_chunks} centroid chunks, got {len(centroid_chunks)}"
    assert sum(batch.shape[0] for batch in data_batches) == n_samples, "Data batches don't sum to original sample count"
    assert sum(chunk.shape[0] for chunk in centroid_chunks) == n_centroids, "Centroid chunks don't sum to original centroid count"
    
    all_assignments = []
    device_idx = 0
    
    # Process each data batch
    for batch_idx, batch_data in enumerate(tqdm(data_batches, desc="Processing validation batches", leave=False)):
        batch_size = batch_data.shape[0]
        
        # For this batch, compute distances to all centroid chunks
        batch_distance_chunks = []
        
        for chunk_idx, centroid_chunk in enumerate(centroid_chunks):
            # Select device in round-robin fashion
            device = devices[device_idx % len(devices)]
            device_idx += 1
            
            # Compute distances for this (batch, centroid_chunk) pair
            distance_chunk = _compute_batch_centroid_distances(
                batch_data, centroid_chunk, device
            )
            batch_distance_chunks.append(distance_chunk)
            
            # Clear device cache after each computation
            if device.type != "cpu":
                if device_type == "cuda":
                    with th.cuda.device(device):
                        backend.empty_cache()
                elif device_type == "xpu":
                    with th.xpu.device(device):
                        backend.empty_cache()
                else:
                    backend.empty_cache()
        
        # Concatenate distance chunks along centroid dimension
        batch_distances = th.cat(batch_distance_chunks, dim=1)
        
        # Sanity check: distances should have shape (batch_size, n_centroids)
        expected_shape = (batch_size, n_centroids)
        assert batch_distances.shape == expected_shape, (
            f"Batch {batch_idx}: Expected distance shape {expected_shape}, "
            f"got {batch_distances.shape}"
        )
        
        # Find closest centroids for this batch
        batch_assignments = th.argmin(batch_distances, dim=1)
        
        # Sanity check: assignments should have shape (batch_size,)
        assert batch_assignments.shape == (batch_size,), (
            f"Batch {batch_idx}: Expected assignment shape ({batch_size},), "
            f"got {batch_assignments.shape}"
        )
        
        all_assignments.append(batch_assignments)
    
    # Final sanity check: total assignments should equal original sample count
    total_assignments = sum(assignments.shape[0] for assignments in all_assignments)
    assert total_assignments == n_samples, (
        f"Total assignments ({total_assignments}) != original samples ({n_samples})"
    )
    
    logger.debug(f"✅ Distributed validation completed: processed {n_samples} samples across {len(devices)} devices")
    
    return all_assignments


def validate_centroid_distribution(
    validation_data: th.Tensor,
    centroids: th.Tensor,
    min_assignment_ratio: float = 0.25,
    max_assignment_ratio: float = 4.0,
    minibatch_size: int = 100000,
    centroid_minibatch_size: int = 65536,
    device_type: DeviceType = "cuda",
) -> tuple[bool, CentroidValidationStats]:
    """
    Validate that centroids produce a reasonable distribution of assignments on validation data.
    
    This function now uses distributed processing across multiple devices, explicitly splitting
    work along both batch and centroid dimensions for better parallelization.

    Args:
        validation_data: Tensor of shape (N, D) containing validation datapoints
        centroids: Tensor of shape (K, D) containing cluster centroids
        min_assignment_ratio: Minimum acceptable ratio (defaults to 0.25)
        max_assignment_ratio: Maximum acceptable ratio (defaults to 4.0)
        minibatch_size: Size of minibatches for device processing (defaults to 100000)
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (defaults to 65536)
        device_type: Device type to use ("cuda" or "xpu", defaults to "cuda")

    Returns:
        Tuple of (is_valid, stats) where:
            - is_valid: True if distribution is reasonable
            - stats: CentroidValidationStats containing distribution statistics
    """
    # Check for any nans
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

    n_samples = validation_data.shape[0]
    logger.debug(
        f"🚀 {device_type.upper()} distributed validation: Processing {n_samples} samples "
        f"in batches of {minibatch_size}, centroids in chunks of {centroid_minibatch_size}"
    )

    # Use distributed processing across devices
    all_assignments = _distribute_work_across_devices(
        validation_data=validation_data,
        centroids=centroids,
        minibatch_size=minibatch_size,
        centroid_minibatch_size=centroid_minibatch_size,
        device_type=device_type,
    )

    # Concatenate all assignments
    assignments = th.cat(all_assignments, dim=0)
    
    # Final sanity check: assignments should match original data size
    assert assignments.shape[0] == n_samples, (
        f"Final assignments shape ({assignments.shape[0]}) != original samples ({n_samples})"
    )

    logger.debug(f"✅ {device_type.upper()} distributed validation completed successfully")

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


def validate_centroid_distribution_legacy(
    validation_data: th.Tensor,
    centroids: th.Tensor,
    min_assignment_ratio: float = 0.25,
    max_assignment_ratio: float = 4.0,
    minibatch_size: int = 100000,
    centroid_minibatch_size: int = 65536,
    device_type: DeviceType = "cuda",
) -> tuple[bool, CentroidValidationStats]:
    """
    Legacy single-device validation function for comparison and fallback.
    
    This is the original implementation that processes everything on a single device.
    Kept for backward compatibility and performance comparison.
    
    Args:
        validation_data: Tensor of shape (N, D) containing validation datapoints
        centroids: Tensor of shape (K, D) containing cluster centroids
        min_assignment_ratio: Minimum acceptable ratio (defaults to 0.25)
        max_assignment_ratio: Maximum acceptable ratio (defaults to 4.0)
        minibatch_size: Size of minibatches for device processing (defaults to 100000)
        centroid_minibatch_size: Size of centroid chunks to avoid device limits (defaults to 65536)
        device_type: Device type to use ("cuda" or "xpu", defaults to "cuda")

    Returns:
        Tuple of (is_valid, stats) where:
            - is_valid: True if distribution is reasonable
            - stats: CentroidValidationStats containing distribution statistics
    """
    # Check for any nans
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

    # Get backend and device based on device_type
    backend = get_backend(device_type)
    device = get_device(device_type) if backend.is_available() else th.device("cpu")

    # Move centroids to device
    centroids_gpu = centroids.to(device).to(th.float32)

    # Process validation data in minibatches to avoid OOM
    all_assignments = []
    n_samples = validation_data.shape[0]

    logger.debug(
        f"🚀 {device_type.upper()} legacy validation: Processing {n_samples} samples in batches of {minibatch_size}, centroids in chunks of {centroid_minibatch_size}"
    )

    for start_idx in tqdm(
        range(0, n_samples, minibatch_size), desc="Legacy validation batches", leave=False
    ):
        end_idx = min(start_idx + minibatch_size, n_samples)
        batch_data = validation_data[start_idx:end_idx].to(device).to(th.float32)

        # Compute distances for this batch, chunking centroids to avoid CUDA limits
        if centroids_gpu.shape[0] <= centroid_minibatch_size:
            # Small enough to compute in one go
            batch_distances = th.cdist(batch_data, centroids_gpu, p=1)
        else:
            # Chunk centroids to avoid device configuration limits
            n_centroids = centroids_gpu.shape[0]
            n_chunks = (
                n_centroids + centroid_minibatch_size - 1
            ) // centroid_minibatch_size

            # Split centroids into chunks
            centroid_chunks = th.tensor_split(centroids_gpu, n_chunks, dim=0)

            # Compute distances for each chunk
            chunk_distances = [
                th.cdist(batch_data, chunk, p=1) for chunk in centroid_chunks
            ]

            # Concatenate along centroid dimension
            batch_distances = th.cat(chunk_distances, dim=1)

        batch_assignments = th.argmin(batch_distances, dim=1)

        # Move back to CPU to save device memory
        all_assignments.append(batch_assignments.cpu())

        # Clear device cache
        del batch_data, batch_distances, batch_assignments
        if device.type != "cpu":
            backend.empty_cache()

    # Concatenate all assignments
    assignments = th.cat(all_assignments, dim=0)

    logger.debug(f"✅ {device_type.upper()} legacy validation completed successfully")

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
