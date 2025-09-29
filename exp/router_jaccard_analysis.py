"""Router Jaccard distance analysis for investigating coactivation patterns in MoE models.

This module implements experiments to analyze expert coactivation using Jaccard distance
as an alternative to correlation analysis. The Jaccard distance measures the similarity
between sets of tokens that activate different experts.
"""

import os
from itertools import count, combinations
from typing import Dict, List, Tuple

import arguably
import numpy as np
import torch as th
from tqdm import tqdm

from exp import OUTPUT_DIR, ACTIVATION_DIRNAME


def compute_jaccard_distance(set_a: set, set_b: set) -> float:
    """Compute Jaccard distance between two sets.
    
    Args:
        set_a: First set of elements
        set_b: Second set of elements
        
    Returns:
        Jaccard distance: 1 - |A ∩ B| / |A ∪ B|
        Returns 0.0 if both sets are empty
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0  # Both sets are empty
    
    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def compute_jaccard_coefficient(set_a: set, set_b: set) -> float:
    """Compute Jaccard coefficient (similarity) between two sets.
    
    Args:
        set_a: First set of elements
        set_b: Second set of elements
        
    Returns:
        Jaccard coefficient: |A ∩ B| / |A ∪ B|
        Returns 0.0 if both sets are empty
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0  # Both sets are empty
    
    return intersection / union


def build_expert_activation_sets(
    router_logits: th.Tensor, 
    top_k: int,
    token_offset: int = 0
) -> Dict[Tuple[int, int], set]:
    """Build sets of token indices that activate each expert.
    
    Args:
        router_logits: Router logits tensor of shape (batch_size, num_layers, num_experts)
        top_k: Number of top experts to consider as activated
        token_offset: Offset to add to token indices (for processing multiple batches)
        
    Returns:
        Dictionary mapping (layer_idx, expert_idx) to set of token indices that activate it
    """
    batch_size, num_layers, num_experts = router_logits.shape
    
    # Get top-k activated experts for each token
    # Shape: (batch_size, num_layers, top_k)
    top_k_indices = th.topk(router_logits, k=top_k, dim=2).indices
    
    # Initialize activation sets for each expert
    expert_activation_sets = {}
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            expert_activation_sets[(layer_idx, expert_idx)] = set()
    
    # Populate activation sets
    for token_idx in range(batch_size):
        global_token_idx = token_offset + token_idx
        for layer_idx in range(num_layers):
            activated_experts = top_k_indices[token_idx, layer_idx]
            for expert_idx in activated_experts:
                expert_key = (layer_idx, expert_idx.item())
                expert_activation_sets[expert_key].add(global_token_idx)
    
    return expert_activation_sets


def analyze_expert_coactivation(
    expert_activation_sets: Dict[Tuple[int, int], set],
    num_layers: int,
    num_experts: int,
    min_activations: int = 10,
) -> Dict:
    """Analyze coactivation patterns between experts using Jaccard distance.
    
    Args:
        expert_activation_sets: Dictionary mapping expert keys to activation sets
        num_layers: Number of layers in the model
        num_experts: Number of experts per layer
        min_activations: Minimum number of activations required to include an expert
        
    Returns:
        Dictionary containing coactivation analysis results
    """
    # Filter experts with sufficient activations
    active_experts = [
        (layer_idx, expert_idx) 
        for (layer_idx, expert_idx), activation_set in expert_activation_sets.items()
        if len(activation_set) >= min_activations
    ]
    
    print(f"Analyzing {len(active_experts)} experts with >= {min_activations} activations")
    
    # Compute pairwise Jaccard distances
    jaccard_distances = []
    jaccard_coefficients = []
    expert_pairs = []
    
    # Within-layer coactivation
    within_layer_distances = []
    within_layer_coefficients = []
    within_layer_pairs = []
    
    # Cross-layer coactivation  
    cross_layer_distances = []
    cross_layer_coefficients = []
    cross_layer_pairs = []
    
    print("Computing pairwise Jaccard distances...")
    for i, expert_a in enumerate(tqdm(active_experts)):
        for expert_b in active_experts[i+1:]:
            layer_a, expert_idx_a = expert_a
            layer_b, expert_idx_b = expert_b
            
            set_a = expert_activation_sets[expert_a]
            set_b = expert_activation_sets[expert_b]
            
            jaccard_dist = compute_jaccard_distance(set_a, set_b)
            jaccard_coeff = compute_jaccard_coefficient(set_a, set_b)
            
            jaccard_distances.append(jaccard_dist)
            jaccard_coefficients.append(jaccard_coeff)
            expert_pairs.append((expert_a, expert_b))
            
            # Categorize by layer relationship
            if layer_a == layer_b:
                within_layer_distances.append(jaccard_dist)
                within_layer_coefficients.append(jaccard_coeff)
                within_layer_pairs.append((expert_a, expert_b))
            else:
                cross_layer_distances.append(jaccard_dist)
                cross_layer_coefficients.append(jaccard_coeff)
                cross_layer_pairs.append((expert_a, expert_b))
    
    # Convert to numpy arrays for analysis
    jaccard_distances = np.array(jaccard_distances)
    jaccard_coefficients = np.array(jaccard_coefficients)
    within_layer_distances = np.array(within_layer_distances)
    within_layer_coefficients = np.array(within_layer_coefficients)
    cross_layer_distances = np.array(cross_layer_distances)
    cross_layer_coefficients = np.array(cross_layer_coefficients)
    
    # Find most similar expert pairs (lowest Jaccard distance / highest coefficient)
    most_similar_indices = np.argsort(jaccard_distances)[:10]
    most_similar_pairs = [
        (expert_pairs[idx], jaccard_distances[idx], jaccard_coefficients[idx])
        for idx in most_similar_indices
    ]
    
    # Find most dissimilar expert pairs (highest Jaccard distance / lowest coefficient)
    most_dissimilar_indices = np.argsort(jaccard_distances)[-10:]
    most_dissimilar_pairs = [
        (expert_pairs[idx], jaccard_distances[idx], jaccard_coefficients[idx])
        for idx in most_dissimilar_indices
    ]
    
    return {
        "total_expert_pairs": len(expert_pairs),
        "active_experts": len(active_experts),
        "within_layer_pairs": len(within_layer_pairs),
        "cross_layer_pairs": len(cross_layer_pairs),
        
        # Overall statistics
        "jaccard_distance_stats": {
            "mean": float(np.mean(jaccard_distances)),
            "std": float(np.std(jaccard_distances)),
            "min": float(np.min(jaccard_distances)),
            "max": float(np.max(jaccard_distances)),
            "median": float(np.median(jaccard_distances)),
        },
        "jaccard_coefficient_stats": {
            "mean": float(np.mean(jaccard_coefficients)),
            "std": float(np.std(jaccard_coefficients)),
            "min": float(np.min(jaccard_coefficients)),
            "max": float(np.max(jaccard_coefficients)),
            "median": float(np.median(jaccard_coefficients)),
        },
        
        # Within-layer statistics
        "within_layer_distance_stats": {
            "mean": float(np.mean(within_layer_distances)) if len(within_layer_distances) > 0 else 0.0,
            "std": float(np.std(within_layer_distances)) if len(within_layer_distances) > 0 else 0.0,
            "min": float(np.min(within_layer_distances)) if len(within_layer_distances) > 0 else 0.0,
            "max": float(np.max(within_layer_distances)) if len(within_layer_distances) > 0 else 0.0,
            "median": float(np.median(within_layer_distances)) if len(within_layer_distances) > 0 else 0.0,
        },
        "within_layer_coefficient_stats": {
            "mean": float(np.mean(within_layer_coefficients)) if len(within_layer_coefficients) > 0 else 0.0,
            "std": float(np.std(within_layer_coefficients)) if len(within_layer_coefficients) > 0 else 0.0,
            "min": float(np.min(within_layer_coefficients)) if len(within_layer_coefficients) > 0 else 0.0,
            "max": float(np.max(within_layer_coefficients)) if len(within_layer_coefficients) > 0 else 0.0,
            "median": float(np.median(within_layer_coefficients)) if len(within_layer_coefficients) > 0 else 0.0,
        },
        
        # Cross-layer statistics
        "cross_layer_distance_stats": {
            "mean": float(np.mean(cross_layer_distances)) if len(cross_layer_distances) > 0 else 0.0,
            "std": float(np.std(cross_layer_distances)) if len(cross_layer_distances) > 0 else 0.0,
            "min": float(np.min(cross_layer_distances)) if len(cross_layer_distances) > 0 else 0.0,
            "max": float(np.max(cross_layer_distances)) if len(cross_layer_distances) > 0 else 0.0,
            "median": float(np.median(cross_layer_distances)) if len(cross_layer_distances) > 0 else 0.0,
        },
        "cross_layer_coefficient_stats": {
            "mean": float(np.mean(cross_layer_coefficients)) if len(cross_layer_coefficients) > 0 else 0.0,
            "std": float(np.std(cross_layer_coefficients)) if len(cross_layer_coefficients) > 0 else 0.0,
            "min": float(np.min(cross_layer_coefficients)) if len(cross_layer_coefficients) > 0 else 0.0,
            "max": float(np.max(cross_layer_coefficients)) if len(cross_layer_coefficients) > 0 else 0.0,
            "median": float(np.median(cross_layer_coefficients)) if len(cross_layer_coefficients) > 0 else 0.0,
        },
        
        # Top similar/dissimilar pairs
        "most_similar_pairs": most_similar_pairs,
        "most_dissimilar_pairs": most_dissimilar_pairs,
    }


@arguably.command()
def analyze_router_jaccard(
    experiment_name: str,
    output_file: str | None = None,
    max_files: int | None = None,
    min_activations: int = 10,
) -> None:
    """Analyze expert coactivation using Jaccard distance.
    
    Args:
        experiment_name: Name of the experiment containing router activations
        output_file: Optional file to save results (defaults to stdout)
        max_files: Maximum number of activation files to process (for testing)
        min_activations: Minimum number of activations required to include an expert
    """
    print(f"Analyzing router coactivation with Jaccard distance for experiment: {experiment_name}")
    
    # Collect expert activation sets across all files
    all_expert_activation_sets = {}
    total_tokens = 0
    top_k: int | None = None
    num_layers: int | None = None
    num_experts: int | None = None
    
    # Process activation files
    files_processed = 0
    for file_idx in tqdm(count(), desc="Processing activation files"):
        if max_files is not None and files_processed >= max_files:
            break
            
        file_path = os.path.join(
            OUTPUT_DIR, experiment_name, ACTIVATION_DIRNAME, f"{file_idx}.pt"
        )
        if not os.path.exists(file_path):
            break
            
        # Load router logits
        try:
            output = th.load(file_path, map_location="cpu")
            if "router_logits" not in output:
                print(f"Warning: No router_logits found in {file_path}")
                continue
                
            router_logits = output["router_logits"]
            top_k = output.get("topk", top_k)
            
            # Get dimensions
            batch_size, num_layers, num_experts = router_logits.shape
            
            # Build activation sets for this batch
            batch_activation_sets = build_expert_activation_sets(
                router_logits, top_k, token_offset=total_tokens
            )
            
            # Merge with global activation sets
            for expert_key, activation_set in batch_activation_sets.items():
                if expert_key not in all_expert_activation_sets:
                    all_expert_activation_sets[expert_key] = set()
                all_expert_activation_sets[expert_key].update(activation_set)
            
            total_tokens += batch_size
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
        files_processed += 1
    
    if total_tokens == 0:
        print("No router logits found in the experiment data!")
        return
    
    print(f"Processed {files_processed} files with {total_tokens} tokens")
    print(f"Model configuration: {num_layers} layers, {num_experts} experts, top-{top_k} routing")
    
    # Analyze coactivation patterns
    results = analyze_expert_coactivation(
        all_expert_activation_sets, num_layers, num_experts, min_activations
    )
    
    # Print results
    print("\n" + "="*60)
    print("ROUTER JACCARD COACTIVATION ANALYSIS")
    print("="*60)
    print(f"Total tokens analyzed: {total_tokens:,}")
    print(f"Active experts (>= {min_activations} activations): {results['active_experts']}")
    print(f"Total expert pairs analyzed: {results['total_expert_pairs']:,}")
    print(f"Within-layer pairs: {results['within_layer_pairs']:,}")
    print(f"Cross-layer pairs: {results['cross_layer_pairs']:,}")
    
    # Overall Jaccard statistics
    print(f"\nOverall Jaccard Distance Statistics:")
    dist_stats = results['jaccard_distance_stats']
    print(f"  Mean: {dist_stats['mean']:.4f} ± {dist_stats['std']:.4f}")
    print(f"  Median: {dist_stats['median']:.4f}")
    print(f"  Range: [{dist_stats['min']:.4f}, {dist_stats['max']:.4f}]")
    
    coeff_stats = results['jaccard_coefficient_stats']
    print(f"\nOverall Jaccard Coefficient Statistics:")
    print(f"  Mean: {coeff_stats['mean']:.4f} ± {coeff_stats['std']:.4f}")
    print(f"  Median: {coeff_stats['median']:.4f}")
    print(f"  Range: [{coeff_stats['min']:.4f}, {coeff_stats['max']:.4f}]")
    
    # Within-layer vs cross-layer comparison
    if results['within_layer_pairs'] > 0:
        within_dist = results['within_layer_distance_stats']
        within_coeff = results['within_layer_coefficient_stats']
        print(f"\nWithin-layer Coactivation:")
        print(f"  Jaccard Distance: {within_dist['mean']:.4f} ± {within_dist['std']:.4f}")
        print(f"  Jaccard Coefficient: {within_coeff['mean']:.4f} ± {within_coeff['std']:.4f}")
    
    if results['cross_layer_pairs'] > 0:
        cross_dist = results['cross_layer_distance_stats']
        cross_coeff = results['cross_layer_coefficient_stats']
        print(f"\nCross-layer Coactivation:")
        print(f"  Jaccard Distance: {cross_dist['mean']:.4f} ± {cross_dist['std']:.4f}")
        print(f"  Jaccard Coefficient: {cross_coeff['mean']:.4f} ± {cross_coeff['std']:.4f}")
    
    # Most similar expert pairs
    print(f"\nMost Similar Expert Pairs (lowest Jaccard distance):")
    for i, ((expert_a, expert_b), distance, coefficient) in enumerate(results['most_similar_pairs'], 1):
        layer_a, expert_a_idx = expert_a
        layer_b, expert_b_idx = expert_b
        layer_type = "within-layer" if layer_a == layer_b else "cross-layer"
        print(f"  {i:2d}. Layer {layer_a} Expert {expert_a_idx} ↔ Layer {layer_b} Expert {expert_b_idx}")
        print(f"      Distance: {distance:.4f}, Coefficient: {coefficient:.4f} ({layer_type})")
    
    # Most dissimilar expert pairs
    print(f"\nMost Dissimilar Expert Pairs (highest Jaccard distance):")
    for i, ((expert_a, expert_b), distance, coefficient) in enumerate(results['most_dissimilar_pairs'], 1):
        layer_a, expert_a_idx = expert_a
        layer_b, expert_b_idx = expert_b
        layer_type = "within-layer" if layer_a == layer_b else "cross-layer"
        print(f"  {i:2d}. Layer {layer_a} Expert {expert_a_idx} ↔ Layer {layer_b} Expert {expert_b_idx}")
        print(f"      Distance: {distance:.4f}, Coefficient: {coefficient:.4f} ({layer_type})")
    
    # Interpretation
    print(f"\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    mean_coeff = coeff_stats['mean']
    if mean_coeff > 0.3:
        coactivation_level = "HIGH"
    elif mean_coeff > 0.1:
        coactivation_level = "MODERATE"
    elif mean_coeff > 0.05:
        coactivation_level = "LOW"
    else:
        coactivation_level = "MINIMAL"
    
    print(f"Expert coactivation level: {coactivation_level}")
    print(f"- Mean Jaccard coefficient of {mean_coeff:.4f} indicates", end=" ")
    
    if coactivation_level == "HIGH":
        print("high overlap in token sets between expert pairs.")
    elif coactivation_level == "MODERATE":
        print("moderate overlap in token sets between expert pairs.")
    elif coactivation_level == "LOW":
        print("low overlap in token sets between expert pairs.")
    else:
        print("minimal overlap in token sets between expert pairs.")
    
    # Compare within-layer vs cross-layer if both exist
    if results['within_layer_pairs'] > 0 and results['cross_layer_pairs'] > 0:
        within_mean = results['within_layer_coefficient_stats']['mean']
        cross_mean = results['cross_layer_coefficient_stats']['mean']
        
        if within_mean > cross_mean * 1.5:
            print(f"- Within-layer coactivation ({within_mean:.4f}) is much higher than cross-layer ({cross_mean:.4f})")
        elif within_mean > cross_mean * 1.1:
            print(f"- Within-layer coactivation ({within_mean:.4f}) is higher than cross-layer ({cross_mean:.4f})")
        elif cross_mean > within_mean * 1.1:
            print(f"- Cross-layer coactivation ({cross_mean:.4f}) is higher than within-layer ({within_mean:.4f})")
        else:
            print(f"- Within-layer and cross-layer coactivation are similar ({within_mean:.4f} vs {cross_mean:.4f})")
    
    # Save results if requested
    if output_file:
        print(f"\nSaving detailed results to {output_file}")
        results_to_save = {
            "experiment_name": experiment_name,
            "model_config": {
                "num_layers": num_layers,
                "num_experts": num_experts,
                "top_k": top_k,
            },
            "analysis_config": {
                "min_activations": min_activations,
                "total_tokens": total_tokens,
            },
            "results": results,
        }
        th.save(results_to_save, output_file)


if __name__ == "__main__":
    arguably.run()
