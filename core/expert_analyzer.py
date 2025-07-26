"""
Expert analysis utilities for studying expert weight patterns and alignment.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ExpertAnalyzer:
    """
    Analyze expert weights and their relationships to routing patterns.
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize expert analyzer.
        
        Args:
            model_wrapper: MoEModelWrapper instance
        """
        self.model_wrapper = model_wrapper
        
    def extract_all_expert_weights(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract expert weight matrices from all layers.
        
        Returns:
            Nested dictionary: {layer_name: {expert_name: {weight_type: tensor}}}
        """
        all_expert_weights = {}
        
        for layer_idx in range(self.model_wrapper.num_layers):
            try:
                expert_weights = self.model_wrapper.get_expert_weights(layer_idx)
                all_expert_weights[f"layer_{layer_idx}"] = expert_weights
            except ValueError as e:
                logger.warning(f"Could not extract expert weights for layer {layer_idx}: {e}")
                
        return all_expert_weights
    
    def compute_expert_similarity_matrix(
        self,
        expert_weights: Dict[str, Dict[str, torch.Tensor]],
        weight_type: str = "gate_proj",
        similarity_metric: str = "cosine"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute similarity matrices between experts within each layer.
        
        Args:
            expert_weights: Dictionary of expert weights
            weight_type: Which weight matrix to analyze ("gate_proj", "up_proj", "down_proj")
            similarity_metric: Similarity metric to use
            
        Returns:
            Dictionary mapping layer names to expert similarity matrices
        """
        similarity_matrices = {}
        
        for layer_name, layer_experts in expert_weights.items():
            expert_names = sorted(layer_experts.keys())
            num_experts = len(expert_names)
            
            if num_experts == 0:
                continue
                
            # Extract weight matrices for this layer
            weight_matrices = []
            for expert_name in expert_names:
                if weight_type in layer_experts[expert_name]:
                    weight_matrix = layer_experts[expert_name][weight_type]
                    # Flatten the weight matrix
                    weight_matrices.append(weight_matrix.flatten())
                    
            if not weight_matrices:
                continue
                
            # Stack into matrix: [num_experts, weight_dim]
            stacked_weights = torch.stack(weight_matrices)
            
            # Compute similarity matrix
            if similarity_metric == "cosine":
                sim_matrix = torch.tensor(
                    cosine_similarity(stacked_weights.cpu().numpy())
                )
            elif similarity_metric == "dot":
                sim_matrix = torch.mm(stacked_weights, stacked_weights.T)
            elif similarity_metric == "l2":
                sim_matrix = -torch.cdist(stacked_weights, stacked_weights, p=2)
            else:
                raise ValueError(f"Unknown similarity metric: {similarity_metric}")
                
            similarity_matrices[layer_name] = sim_matrix
            
        return similarity_matrices
    
    def analyze_expert_router_alignment(
        self,
        expert_weights: Dict[str, Dict[str, torch.Tensor]],
        router_weights: Dict[str, torch.Tensor],
        weight_type: str = "gate_proj"
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze alignment between expert weights and router directions.
        
        Args:
            expert_weights: Dictionary of expert weights
            router_weights: Dictionary of router weights
            weight_type: Which expert weight matrix to analyze
            
        Returns:
            Dictionary mapping layer names to alignment matrices
        """
        alignment_results = {}
        
        for layer_name in expert_weights.keys():
            if layer_name not in router_weights:
                continue
                
            layer_experts = expert_weights[layer_name]
            layer_router = router_weights[layer_name]  # [num_experts, hidden_dim]
            
            expert_names = sorted(layer_experts.keys())
            alignments = []
            
            for i, expert_name in enumerate(expert_names):
                if weight_type not in layer_experts[expert_name]:
                    continue
                    
                expert_weight = layer_experts[expert_name][weight_type]  # [out_dim, in_dim]
                router_vector = layer_router[i]  # [hidden_dim]
                
                # Compute alignment between router vector and expert weight subspace
                alignment = self._compute_router_expert_alignment(
                    router_vector, expert_weight
                )
                alignments.append(alignment)
                
            if alignments:
                alignment_results[layer_name] = torch.tensor(alignments)
                
        return alignment_results
    
    def _compute_router_expert_alignment(
        self,
        router_vector: torch.Tensor,
        expert_weight: torch.Tensor
    ) -> float:
        """
        Compute alignment between router vector and expert weight matrix.
        
        Args:
            router_vector: Router vector [hidden_dim]
            expert_weight: Expert weight matrix [out_dim, in_dim]
            
        Returns:
            Alignment score
        """
        # Method 1: Project router vector onto expert input space
        if expert_weight.shape[1] == router_vector.shape[0]:
            # Router vector aligns with input dimension
            projection = torch.mv(expert_weight, router_vector)  # [out_dim]
            alignment = torch.norm(projection).item()
        else:
            # Method 2: Compute similarity with principal components of expert weights
            U, S, Vt = torch.svd(expert_weight)
            # Project router vector onto top principal component
            if Vt.shape[1] == router_vector.shape[0]:
                top_component = Vt[0]  # [in_dim]
                alignment = torch.dot(router_vector, top_component).abs().item()
            else:
                # Fallback: use Frobenius norm similarity
                router_expanded = router_vector.unsqueeze(0).expand_as(expert_weight)
                alignment = torch.cosine_similarity(
                    expert_weight.flatten(), 
                    router_expanded.flatten(), 
                    dim=0
                ).item()
                
        return alignment
    
    def analyze_expert_specialization_patterns(
        self,
        expert_weights: Dict[str, Dict[str, torch.Tensor]],
        router_logits: Dict[str, torch.Tensor],
        data_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how expert weights relate to their usage patterns.
        
        Args:
            expert_weights: Dictionary of expert weights
            router_logits: Dictionary of router logits
            data_labels: Optional data type labels
            
        Returns:
            Dictionary containing specialization analysis
        """
        specialization_results = {}
        
        if data_labels is None:
            data_labels = ["general"] * list(router_logits.values())[0].shape[0]
            
        unique_labels = list(set(data_labels))
        
        for layer_name in expert_weights.keys():
            if layer_name.replace("layer_", "layer_") + "_router" not in router_logits:
                continue
                
            router_key = layer_name.replace("layer_", "layer_") + "_router"
            layer_router_logits = router_logits[router_key]
            layer_experts = expert_weights[layer_name]
            
            # Compute expert usage by data type
            expert_usage_by_type = {}
            router_probs = torch.softmax(layer_router_logits, dim=-1)
            
            for label in unique_labels:
                mask = [i for i, l in enumerate(data_labels) if l == label]
                if mask:
                    expert_usage_by_type[label] = torch.mean(router_probs[mask], dim=0)
            
            # Analyze weight patterns for specialized vs general experts
            expert_names = sorted(layer_experts.keys())
            weight_analysis = {}
            
            for i, expert_name in enumerate(expert_names):
                expert_data = layer_experts[expert_name]
                
                # Compute weight statistics
                weight_stats = {}
                for weight_type, weight_matrix in expert_data.items():
                    weight_stats[weight_type] = {
                        "norm": torch.norm(weight_matrix).item(),
                        "sparsity": (weight_matrix.abs() < 1e-6).float().mean().item(),
                        "max_activation": torch.max(weight_matrix).item(),
                        "min_activation": torch.min(weight_matrix).item(),
                    }
                
                # Compute specialization score based on usage variance across data types
                usage_scores = [expert_usage_by_type[label][i].item() for label in unique_labels]
                specialization_score = np.std(usage_scores) / (np.mean(usage_scores) + 1e-8)
                
                weight_analysis[expert_name] = {
                    "weight_stats": weight_stats,
                    "usage_by_type": {label: expert_usage_by_type[label][i].item() for label in unique_labels},
                    "specialization_score": specialization_score,
                }
            
            specialization_results[layer_name] = weight_analysis
            
        return specialization_results
    
    def compute_cross_layer_expert_similarity(
        self,
        expert_weights: Dict[str, Dict[str, torch.Tensor]],
        weight_type: str = "gate_proj"
    ) -> torch.Tensor:
        """
        Compute similarity between experts across different layers.
        
        Args:
            expert_weights: Dictionary of expert weights
            weight_type: Which weight matrix to analyze
            
        Returns:
            Cross-layer expert similarity tensor
        """
        layer_names = sorted(expert_weights.keys())
        num_layers = len(layer_names)
        
        if num_layers == 0:
            return torch.empty(0)
        
        # Get number of experts (assuming consistent across layers)
        first_layer = expert_weights[layer_names[0]]
        num_experts = len(first_layer)
        
        # Initialize similarity tensor: [num_layers, num_layers, num_experts, num_experts]
        similarity_tensor = torch.zeros(num_layers, num_layers, num_experts, num_experts)
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                experts_i = expert_weights[layer_i]
                experts_j = expert_weights[layer_j]
                
                expert_names_i = sorted(experts_i.keys())
                expert_names_j = sorted(experts_j.keys())
                
                for ei, expert_i in enumerate(expert_names_i):
                    for ej, expert_j in enumerate(expert_names_j):
                        if weight_type in experts_i[expert_i] and weight_type in experts_j[expert_j]:
                            weight_i = experts_i[expert_i][weight_type].flatten()
                            weight_j = experts_j[expert_j][weight_type].flatten()
                            
                            # Compute cosine similarity
                            similarity = torch.cosine_similarity(
                                weight_i.unsqueeze(0), 
                                weight_j.unsqueeze(0)
                            ).item()
                            
                            similarity_tensor[i, j, ei, ej] = similarity
        
        return similarity_tensor
    
    def analyze_expert_subspaces(
        self,
        expert_weights: Dict[str, Dict[str, torch.Tensor]],
        weight_type: str = "gate_proj",
        n_components: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze subspace structure of expert weights.
        
        Args:
            expert_weights: Dictionary of expert weights
            weight_type: Which weight matrix to analyze
            n_components: Number of components for dimensionality reduction
            
        Returns:
            Dictionary containing subspace analysis results
        """
        subspace_results = {}
        
        for layer_name, layer_experts in expert_weights.items():
            expert_names = sorted(layer_experts.keys())
            
            # Collect weight matrices
            weight_matrices = []
            for expert_name in expert_names:
                if weight_type in layer_experts[expert_name]:
                    weight_matrix = layer_experts[expert_name][weight_type]
                    weight_matrices.append(weight_matrix.flatten())
            
            if not weight_matrices:
                continue
                
            # Stack weights: [num_experts, weight_dim]
            stacked_weights = torch.stack(weight_matrices)
            
            # Perform SVD
            U, S, Vt = torch.svd(stacked_weights)
            
            # PCA analysis
            pca = PCA(n_components=min(n_components, stacked_weights.shape[0]))
            pca.fit(stacked_weights.cpu().numpy())
            
            # Compute explained variance
            explained_variance = S**2
            explained_variance_ratio = explained_variance / explained_variance.sum()
            
            subspace_results[layer_name] = {
                "singular_values": S,
                "explained_variance_ratio": explained_variance_ratio,
                "principal_components": Vt[:n_components],
                "pca_explained_variance_ratio": torch.tensor(pca.explained_variance_ratio_),
                "effective_rank": torch.sum(explained_variance_ratio > 0.01).item(),
                "weight_matrices": stacked_weights,  # For further analysis
            }
            
        return subspace_results

