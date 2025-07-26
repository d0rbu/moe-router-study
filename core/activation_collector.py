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
        prompts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect router logits and gating decisions.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing router logits per layer
        """
        logger.info(f"Collecting router activations for {len(prompts)} prompts")
        
        router_data = {}
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting router data"):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.model_wrapper.tokenize(
                batch_prompts, 
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.nn_model.device) for k, v in inputs.items()}
            
            # Forward pass with router logits
            with torch.no_grad():
                logits, router_logits = self.model_wrapper.forward_with_router_logits(
                    inputs["input_ids"],
                    inputs.get("attention_mask")
                )
            
            # Store router logits per layer
            for layer_idx, layer_router_logits in enumerate(router_logits):
                layer_key = f"layer_{layer_idx}_router"
                
                if layer_key not in router_data:
                    router_data[layer_key] = []
                    
                # Extract last token router logits
                last_token_logits = layer_router_logits[:, -1, :]  # [batch, num_experts]
                router_data[layer_key].append(last_token_logits.cpu())
        
        # Concatenate batches
        for key in router_data:
            router_data[key] = torch.cat(router_data[key], dim=0)
            
        return router_data
    
    def collect_expert_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect individual expert activations for a specific layer.
        This requires custom tracing since nnterp doesn't expose expert-level outputs.
        
        Args:
            prompts: List of input prompts
            layer_idx: Layer to analyze
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing expert activations
        """
        logger.info(f"Collecting expert activations for layer {layer_idx}")
        
        expert_activations = {}
        
        # This would require custom hooks or model modification
        # For now, we'll collect the combined MLP output and router decisions
        # Individual expert outputs would need more complex tracing
        
        # Get layer activations and router logits
        layer_acts = self.collect_layer_activations(
            prompts, 
            layers=[layer_idx], 
            batch_size=batch_size,
            max_length=max_length
        )
        
        router_acts = self.collect_router_activations(
            prompts,
            batch_size=batch_size, 
            max_length=max_length
        )
        
        expert_activations[f"layer_{layer_idx}_mlp_output"] = layer_acts[f"layer_{layer_idx}"]
        expert_activations[f"layer_{layer_idx}_router_logits"] = router_acts[f"layer_{layer_idx}_router"]
        
        return expert_activations
    
    def compute_activation_correlations(
        self,
        activations: Dict[str, torch.Tensor],
        method: str = "pearson"
    ) -> torch.Tensor:
        """
        Compute correlations between layer activations.
        
        Args:
            activations: Dictionary of layer activations
            method: Correlation method ("pearson", "spearman")
            
        Returns:
            Correlation matrix between layers
        """
        layer_keys = sorted([k for k in activations.keys() if k.startswith("layer_")])
        num_layers = len(layer_keys)
        
        # Stack activations: [num_layers, num_samples, hidden_dim]
        stacked_acts = torch.stack([activations[key] for key in layer_keys])
        
        # Flatten to [num_layers, num_samples * hidden_dim]
        flattened_acts = stacked_acts.flatten(start_dim=1)
        
        if method == "pearson":
            # Compute Pearson correlation
            corr_matrix = torch.corrcoef(flattened_acts)
        else:
            raise NotImplementedError(f"Correlation method {method} not implemented")
            
        return corr_matrix
    
    def save_activations(
        self,
        activations: Dict[str, torch.Tensor],
        filepath: str,
        compress: bool = True
    ):
        """Save activations to disk."""
        save_dict = {k: v.cpu().numpy() for k, v in activations.items()}
        
        if compress:
            np.savez_compressed(filepath, **save_dict)
        else:
            np.savez(filepath, **save_dict)
            
        logger.info(f"Saved activations to {filepath}")
    
    def load_activations(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load activations from disk."""
        data = np.load(filepath)
        activations = {k: torch.from_numpy(data[k]) for k in data.keys()}
        
        logger.info(f"Loaded activations from {filepath}")
        return activations

