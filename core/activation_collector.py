"""
Activation collection utilities for MoE models using nnterp.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from nnterp import collect_activations_batched
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ActivationCollector:
    """
    Collect and manage activations from MoE models across different datasets.
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize activation collector.
        
        Args:
            model_wrapper: MoEModelWrapper instance
        """
        self.model_wrapper = model_wrapper
        self.nn_model = model_wrapper.nn_model
        
    def collect_layer_activations(
        self,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        batch_size: int = 8,
        token_idx: int = -1,  # Last token by default
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect layer-wise activations for given prompts.
        
        Args:
            prompts: List of input prompts
            layers: Specific layers to collect (None for all)
            batch_size: Batch size for processing
            token_idx: Token position to extract (-1 for last)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if layers is None:
            layers = list(range(self.model_wrapper.num_layers))
            
        logger.info(f"Collecting activations for {len(prompts)} prompts across {len(layers)} layers")
        
        # Use nnterp's batched collection
        activations = collect_activations_batched(
            self.nn_model,
            prompts,
            batch_size=batch_size,
            layers=layers,
            idx=token_idx,
            max_length=max_length,
        )
        
        # Convert to dictionary format
        activation_dict = {}
        for i, layer_idx in enumerate(layers):
            activation_dict[f"layer_{layer_idx}"] = activations[i]
            
        return activation_dict
    
    def collect_router_activations(
        self,
        layer_activations: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute router logits from layer activations using router weights.
        
        Args:
            layer_activations: Dictionary mapping layer names to activation tensors
            temperature: Temperature for softmax computation
            
        Returns:
            Dictionary containing router logits and probabilities per layer
        """
        logger.info(f"Computing router activations for {len(layer_activations)} layers")
        
        router_data = {}
        
        for layer_name, activations in layer_activations.items():
            # Extract layer index from layer name (e.g., "layer_5" -> 5)
            layer_idx = int(layer_name.split("_")[1])
            
            try:
                # Get router weights for this layer
                router_weights = self.model_wrapper.get_router_weights(layer_idx)  # [num_experts, hidden_dim]
                
                # Compute router logits: activations @ router_weights.T
                # activations: [batch_size, hidden_dim]
                # router_weights: [num_experts, hidden_dim]
                router_logits = torch.mm(activations, router_weights.T)  # [batch_size, num_experts]
                
                # Apply temperature and compute probabilities
                router_probs = torch.softmax(router_logits / temperature, dim=-1)
                
                # Store results
                router_data[f"{layer_name}_router_logits"] = router_logits
                router_data[f"{layer_name}_router_probs"] = router_probs
                
            except ValueError as e:
                logger.warning(f"Could not compute router activations for {layer_name}: {e}")
                continue
        
        return router_data
    
    def collect_expert_router_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect expert and router activations for a specific layer.
        
        Args:
            prompts: List of input prompts
            layer_idx: Layer to analyze
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing expert and router activations
        """
        logger.info(f"Collecting expert and router activations for layer {layer_idx}")
        
        expert_router_activations = {}
        
        # Get layer activations
        layer_acts = self.collect_layer_activations(
            prompts, 
            layers=[layer_idx], 
            batch_size=batch_size,
            max_length=max_length
        )
        
        # Compute router activations from layer activations
        router_acts = self.collect_router_activations(layer_acts)
        
        expert_router_activations[f"layer_{layer_idx}_mlp_output"] = layer_acts[f"layer_{layer_idx}"]
        expert_router_activations[f"layer_{layer_idx}_router_logits"] = router_acts[f"layer_{layer_idx}_router_logits"]
        expert_router_activations[f"layer_{layer_idx}_router_probs"] = router_acts[f"layer_{layer_idx}_router_probs"]
        
        return expert_router_activations
