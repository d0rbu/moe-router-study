"""
SVD-based circuit discovery for router models.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

from exp.activations import get_activation_filepaths, get_router_logits


def discover_circuits_svd(
    experiment_name: str,
    n_components: int = 10,
    layer_idx: Optional[int] = None,
    random_state: int = 42,
) -> Dict[int, Dict[str, Union[np.ndarray, List[float]]]]:
    """
    Discover circuits using SVD on router logits.

    Args:
        experiment_name: Name of the experiment
        n_components: Number of SVD components to extract
        layer_idx: Layer index to analyze (if None, analyze all layers)
        random_state: Random seed for SVD

    Returns:
        Dictionary mapping layer indices to dictionaries containing:
            - 'components': SVD components (n_components, n_experts)
            - 'singular_values': Singular values
            - 'explained_variance_ratio': Explained variance ratio
    """
    # Get activation filepaths
    activation_filepaths = get_activation_filepaths(experiment_name, "router_logits")
    
    # Get router logits
    router_logits_by_layer = get_router_logits(activation_filepaths)
    
    # If layer_idx is specified, only analyze that layer
    if layer_idx is not None:
        if layer_idx not in router_logits_by_layer:
            raise ValueError(f"Layer {layer_idx} not found in router logits")
        layers_to_analyze = {layer_idx: router_logits_by_layer[layer_idx]}
    else:
        layers_to_analyze = router_logits_by_layer
    
    # Perform SVD on each layer
    results = {}
    for layer_idx, router_logits in tqdm(layers_to_analyze.items(), desc="SVD analysis"):
        # Reshape to (n_tokens, n_experts)
        n_tokens = router_logits.shape[0] * router_logits.shape[1]
        n_experts = router_logits.shape[2]
        reshaped_logits = router_logits.reshape(n_tokens, n_experts)
        
        # Convert to numpy for sklearn
        if isinstance(reshaped_logits, th.Tensor):
            reshaped_logits = reshaped_logits.numpy()
        
        # Perform SVD
        svd = TruncatedSVD(n_components=min(n_components, n_experts - 1), random_state=random_state)
        svd.fit(reshaped_logits)
        
        # Store results
        results[layer_idx] = {
            'components': svd.components_,
            'singular_values': svd.singular_values_,
            'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
        }
    
    return results


def get_circuit_experts(
    components: np.ndarray,
    threshold: float = 0.2,
) -> List[List[int]]:
    """
    Extract expert indices for each circuit based on SVD components.

    Args:
        components: SVD components (n_components, n_experts)
        threshold: Threshold for including an expert in a circuit

    Returns:
        List of lists, where each inner list contains expert indices for a circuit
    """
    circuits = []
    for component_idx in range(components.shape[0]):
        component = components[component_idx]
        
        # Normalize component
        component = component / np.linalg.norm(component)
        
        # Find experts with absolute value above threshold
        circuit_experts = np.where(np.abs(component) > threshold)[0].tolist()
        circuits.append(circuit_experts)
    
    return circuits


def analyze_circuits(
    experiment_name: str,
    n_components: int = 10,
    threshold: float = 0.2,
    layer_idx: Optional[int] = None,
    random_state: int = 42,
) -> Dict[int, Dict[str, Union[np.ndarray, List[float], List[List[int]]]]]:
    """
    Analyze circuits using SVD and extract expert indices.

    Args:
        experiment_name: Name of the experiment
        n_components: Number of SVD components to extract
        threshold: Threshold for including an expert in a circuit
        layer_idx: Layer index to analyze (if None, analyze all layers)
        random_state: Random seed for SVD

    Returns:
        Dictionary mapping layer indices to dictionaries containing:
            - 'components': SVD components (n_components, n_experts)
            - 'singular_values': Singular values
            - 'explained_variance_ratio': Explained variance ratio
            - 'circuits': List of lists, where each inner list contains expert indices for a circuit
    """
    # Discover circuits using SVD
    svd_results = discover_circuits_svd(
        experiment_name=experiment_name,
        n_components=n_components,
        layer_idx=layer_idx,
        random_state=random_state,
    )
    
    # Extract expert indices for each circuit
    for layer_idx, layer_results in svd_results.items():
        components = layer_results['components']
        circuits = get_circuit_experts(components, threshold=threshold)
        layer_results['circuits'] = circuits
    
    return svd_results

