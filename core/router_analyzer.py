"""
Router analysis utilities for studying MoE routing patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class RouterAnalyzer:
    """
    Analyze router weights and routing patterns in MoE models.
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize router analyzer.
        
        Args:
            model_wrapper: MoEModelWrapper instance
        """
        self.model_wrapper = model_wrapper
        
    def extract_all_router_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract router weight matrices from all layers.
        
        Returns:
            Dictionary mapping layer names to router weights
        """
        router_weights = {}
        
        for layer_idx in range(self.model_wrapper.num_layers):
            try:
                weights = self.model_wrapper.get_router_weights(layer_idx)
                router_weights[f"layer_{layer_idx}"] = weights
            except ValueError as e:
                logger.warning(f"Could not extract router weights for layer {layer_idx}: {e}")
                
        return router_weights
    
    def compute_router_similarity_matrix(
        self,
        router_weights: Dict[str, torch.Tensor],
        similarity_metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute similarity matrix between router weight vectors.
        
        Args:
            router_weights: Dictionary of router weights per layer
            similarity_metric: Similarity metric ("cosine", "dot", "l2")
            
        Returns:
            Similarity matrix [num_layers, num_layers, num_experts, num_experts]
        """
        layer_keys = sorted(router_weights.keys())
        num_layers = len(layer_keys)
        
        if num_layers == 0:
            return torch.empty(0)
            
        # Get number of experts from first layer
        first_layer_weights = router_weights[layer_keys[0]]
        num_experts = first_layer_weights.shape[0]
        
        # Initialize similarity tensor
        similarity_matrix = torch.zeros(num_layers, num_layers, num_experts, num_experts)
        
        for i, layer_i in enumerate(layer_keys):
            for j, layer_j in enumerate(layer_keys):
                weights_i = router_weights[layer_i]  # [num_experts, hidden_dim]
                weights_j = router_weights[layer_j]  # [num_experts, hidden_dim]
                
                if similarity_metric == "cosine":
                    # Compute cosine similarity between all expert pairs
                    sim_matrix = torch.tensor(
                        cosine_similarity(weights_i.cpu().numpy(), weights_j.cpu().numpy())
                    )
                elif similarity_metric == "dot":
                    # Dot product similarity
                    sim_matrix = torch.mm(weights_i, weights_j.T)
                elif similarity_metric == "l2":
                    # Negative L2 distance (higher = more similar)
                    sim_matrix = -torch.cdist(weights_i, weights_j, p=2)
                else:
                    raise ValueError(f"Unknown similarity metric: {similarity_metric}")
                    
                similarity_matrix[i, j] = sim_matrix
                
        return similarity_matrix
    
    def analyze_router_subspaces(
        self,
        router_weights: Dict[str, torch.Tensor],
        n_components: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the subspace structure of router weights using SVD/PCA.
        
        Args:
            router_weights: Dictionary of router weights per layer
            n_components: Number of principal components to analyze
            
        Returns:
            Dictionary containing subspace analysis results
        """
        subspace_analysis = {}
        
        for layer_name, weights in router_weights.items():
            # weights: [num_experts, hidden_dim]
            
            # Perform SVD
            U, S, Vt = torch.svd(weights)
            
            # Compute explained variance ratio
            explained_variance = S**2
            explained_variance_ratio = explained_variance / explained_variance.sum()
            
            # PCA for comparison
            pca = PCA(n_components=min(n_components, weights.shape[0]))
            pca.fit(weights.cpu().numpy())
            
            subspace_analysis[layer_name] = {
                "singular_values": S,
                "explained_variance_ratio": explained_variance_ratio,
                "principal_components": Vt[:n_components],  # Top n components
                "pca_components": torch.tensor(pca.components_),
                "pca_explained_variance_ratio": torch.tensor(pca.explained_variance_ratio_),
                "effective_rank": torch.sum(explained_variance_ratio > 0.01).item(),  # Components explaining >1% variance
            }
            
        return subspace_analysis
    
    def compute_cross_layer_alignment(
        self,
        router_weights: Dict[str, torch.Tensor],
        method: str = "principal_angles"
    ) -> torch.Tensor:
        """
        Compute alignment between router subspaces across layers.
        
        Args:
            router_weights: Dictionary of router weights per layer
            method: Alignment method ("principal_angles", "subspace_overlap")
            
        Returns:
            Cross-layer alignment matrix
        """
        layer_keys = sorted(router_weights.keys())
        num_layers = len(layer_keys)
        alignment_matrix = torch.zeros(num_layers, num_layers)
        
        for i, layer_i in enumerate(layer_keys):
            for j, layer_j in enumerate(layer_keys):
                weights_i = router_weights[layer_i]
                weights_j = router_weights[layer_j]
                
                if method == "principal_angles":
                    # Compute principal angles between subspaces
                    alignment = self._compute_principal_angles(weights_i, weights_j)
                elif method == "subspace_overlap":
                    # Compute subspace overlap using Frobenius norm
                    alignment = self._compute_subspace_overlap(weights_i, weights_j)
                else:
                    raise ValueError(f"Unknown alignment method: {method}")
                    
                alignment_matrix[i, j] = alignment
                
        return alignment_matrix
    
    def _compute_principal_angles(
        self,
        weights_i: torch.Tensor,
        weights_j: torch.Tensor
    ) -> float:
        """Compute principal angles between two subspaces."""
        # Orthogonalize the weight matrices
        Q_i, _ = torch.qr(weights_i.T)  # [hidden_dim, num_experts]
        Q_j, _ = torch.qr(weights_j.T)  # [hidden_dim, num_experts]
        
        # Compute SVD of Q_i^T @ Q_j
        _, S, _ = torch.svd(torch.mm(Q_i.T, Q_j))
        
        # Principal angles are arccos of singular values
        # Return mean cosine of principal angles as alignment measure
        return torch.mean(S).item()
    
    def _compute_subspace_overlap(
        self,
        weights_i: torch.Tensor,
        weights_j: torch.Tensor
    ) -> float:
        """Compute subspace overlap using projection."""
        # Normalize weights
        weights_i_norm = weights_i / torch.norm(weights_i, dim=1, keepdim=True)
        weights_j_norm = weights_j / torch.norm(weights_j, dim=1, keepdim=True)
        
        # Compute Gram matrices
        gram_i = torch.mm(weights_i_norm, weights_i_norm.T)
        gram_j = torch.mm(weights_j_norm, weights_j_norm.T)
        
        # Frobenius norm of difference
        overlap = torch.norm(gram_i - gram_j, p='fro').item()
        
        # Convert to similarity (lower difference = higher similarity)
        return 1.0 / (1.0 + overlap)
    
    def analyze_routing_patterns(
        self,
        router_logits: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze routing patterns from router logits.
        
        Args:
            router_logits: Dictionary of router logits per layer
            temperature: Temperature for softmax (lower = more concentrated)
            
        Returns:
            Dictionary containing routing pattern analysis
        """
        routing_analysis = {}
        
        for layer_name, logits in router_logits.items():
            # logits: [num_samples, num_experts]
            
            # Apply temperature and softmax
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # Compute routing statistics
            expert_usage = torch.mean(probs, dim=0)  # Average usage per expert
            routing_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # Per-sample entropy
            
            # Compute load balancing
            load_balance = torch.std(expert_usage) / torch.mean(expert_usage)
            
            # Top-k routing analysis
            top_experts = torch.topk(probs, k=min(3, probs.shape[-1]), dim=-1)
            
            routing_analysis[layer_name] = {
                "expert_usage": expert_usage,
                "mean_routing_entropy": torch.mean(routing_entropy).item(),
                "std_routing_entropy": torch.std(routing_entropy).item(),
                "load_balance_coefficient": load_balance.item(),
                "top_expert_indices": top_experts.indices,
                "top_expert_probs": top_experts.values,
                "routing_concentration": torch.mean(torch.max(probs, dim=-1)[0]).item(),  # How concentrated routing is
            }
            
        return routing_analysis
    
    def compute_expert_specialization(
        self,
        router_logits: Dict[str, torch.Tensor],
        data_labels: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute expert specialization patterns across different data types.
        
        Args:
            router_logits: Dictionary of router logits per layer
            data_labels: Optional labels for data types (e.g., "math", "code", "general")
            
        Returns:
            Dictionary containing specialization analysis
        """
        if data_labels is None:
            data_labels = ["general"] * list(router_logits.values())[0].shape[0]
            
        specialization_analysis = {}
        unique_labels = list(set(data_labels))
        
        for layer_name, logits in router_logits.items():
            probs = torch.softmax(logits, dim=-1)
            
            # Compute expert usage by data type
            expert_usage_by_type = {}
            for label in unique_labels:
                mask = [i for i, l in enumerate(data_labels) if l == label]
                if mask:
                    expert_usage_by_type[label] = torch.mean(probs[mask], dim=0)
                    
            specialization_analysis[layer_name] = expert_usage_by_type
            
        return specialization_analysis

