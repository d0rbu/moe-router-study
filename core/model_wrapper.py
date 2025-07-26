"""
Wrapper for MoE models using nnterp with enhanced MoE-specific functionality.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from nnterp import load_model
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class MoEModelWrapper:
    """
    Enhanced wrapper for MoE models that provides easy access to:
    - Router logits and gating decisions
    - Expert activations and weights
    - Layer-wise hidden states
    - Checkpoint loading for training history analysis
    """
    
    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0924",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
    ):
        """
        Initialize MoE model wrapper.
        
        Args:
            model_name: HuggingFace model name or path
            device_map: Device mapping strategy
            torch_dtype: Model precision
            trust_remote_code: Whether to trust remote code
            revision: Specific model revision/checkpoint
        """
        self.model_name = model_name
        self.revision = revision
        
        logger.info(f"Loading model: {model_name}")
        if revision:
            logger.info(f"Using revision: {revision}")
            
        # Load model with nnterp
        self.nn_model = load_model(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        
        self.tokenizer = self.nn_model.tokenizer
        self.model = self.nn_model.model  # Access underlying HF model
        
        # Cache model architecture info
        self._analyze_architecture()
        
    def _analyze_architecture(self):
        """Analyze and cache MoE architecture details."""
        config = self.model.config
        
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        
        # MoE-specific attributes (may vary by model)
        self.num_experts = getattr(config, 'num_experts', None)
        self.num_experts_per_tok = getattr(config, 'num_experts_per_tok', None)
        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', None)
        
        logger.info(f"Model architecture:")
        logger.info(f"  Layers: {self.num_layers}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Experts per layer: {self.num_experts}")
        logger.info(f"  Experts per token: {self.num_experts_per_tok}")
        
    def get_expert_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract expert weight matrices for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dictionary containing expert weights (gate_proj, up_proj, down_proj)
        """
        layer = self.model.layers[layer_idx]
        
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts'):
            raise ValueError(f"Layer {layer_idx} doesn't appear to be a MoE layer")
            
        expert_weights = {}
        
        for expert_idx, expert in enumerate(layer.mlp.experts):
            expert_weights[f"expert_{expert_idx}"] = {
                "gate_proj": expert.gate_proj.weight.data.clone(),
                "up_proj": expert.up_proj.weight.data.clone(), 
                "down_proj": expert.down_proj.weight.data.clone(),
            }
            
        return expert_weights
    
    def get_router_weights(self, layer_idx: int) -> torch.Tensor:
        """
        Extract router/gating weights for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Router weight matrix
        """
        layer = self.model.layers[layer_idx]
        
        if hasattr(layer.mlp, 'gate'):
            return layer.mlp.gate.weight.data.clone()
        elif hasattr(layer.mlp, 'router'):
            return layer.mlp.router.weight.data.clone()
        else:
            raise ValueError(f"Cannot find router weights in layer {layer_idx}")
    
    def forward_with_router_logits(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns both model outputs and router logits.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (model_outputs, router_logits_per_layer)
        """
        # Enable router logits output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
            return_dict=True,
        )
        
        return outputs.logits, outputs.router_logits
    
    def get_layer_names(self) -> List[str]:
        """Get standardized layer names for nnterp."""
        return [f"layers.{i}" for i in range(self.num_layers)]
    
    def tokenize(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs
        )
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        """Decode token IDs back to text."""
        return self.tokenizer.batch_decode(token_ids, **kwargs)
    
    @classmethod
    def load_checkpoint(
        cls,
        model_name: str = "allenai/OLMoE-1B-7B-0924", 
        step: Optional[int] = None,
        **kwargs
    ) -> "MoEModelWrapper":
        """
        Load a specific training checkpoint.
        
        Args:
            model_name: Base model name
            step: Training step (if available)
            **kwargs: Additional arguments for model loading
            
        Returns:
            MoEModelWrapper instance
        """
        # For OLMoE, checkpoints might be available as revisions
        # This would need to be adapted based on actual checkpoint availability
        revision = f"step_{step}" if step else None
        
        return cls(
            model_name=model_name,
            revision=revision,
            **kwargs
        )

