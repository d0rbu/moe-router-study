"""
Basic routing analysis experiment - collect and analyze routing patterns.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import wandb
from typing import Dict, List, Any

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import MoEModelWrapper, ActivationCollector, RouterAnalyzer
from data import DatasetLoader, PromptGenerator
from viz.plotting import RouterPlotter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicRoutingAnalysis:
    """
    Basic experiment to analyze routing patterns in MoE models.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/OLMoE-1B-7B-0924",
        output_dir: str = "results/basic_routing",
        use_wandb: bool = True,
        wandb_project: str = "moe-routing-study"
    ):
        """
        Initialize the experiment.
        
        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save results
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name="basic_routing_analysis",
                config={
                    "model_name": model_name,
                    "experiment": "basic_routing_analysis"
                }
            )
        
        # Initialize components
        logger.info("Loading model...")
        self.model_wrapper = MoEModelWrapper(model_name)
        self.activation_collector = ActivationCollector(self.model_wrapper)
        self.router_analyzer = RouterAnalyzer(self.model_wrapper)
        self.dataset_loader = DatasetLoader()
        self.prompt_generator = PromptGenerator()
        self.plotter = RouterPlotter()
        
    def run_experiment(self):
        """Run the complete routing analysis experiment."""
        logger.info("Starting basic routing analysis experiment")
        
        # Step 1: Load datasets
        logger.info("Loading datasets...")
        datasets = self._load_datasets()
        
        # Step 2: Collect router activations
        logger.info("Collecting router activations...")
        router_activations = self._collect_router_activations(datasets)
        
        # Step 3: Analyze router weights
        logger.info("Analyzing router weights...")
        router_weights = self._analyze_router_weights()
        
        # Step 4: Compute routing patterns
        logger.info("Computing routing patterns...")
        routing_patterns = self._analyze_routing_patterns(router_activations)
        
        # Step 5: Analyze correlations
        logger.info("Analyzing cross-layer correlations...")
        correlations = self._analyze_correlations(router_activations)
        
        # Step 6: Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_visualizations(
            router_activations, 
            router_weights, 
            routing_patterns, 
            correlations
        )
        
        # Step 7: Save results
        logger.info("Saving results...")
        self._save_results({
            "router_activations": router_activations,
            "router_weights": router_weights,
            "routing_patterns": routing_patterns,
            "correlations": correlations,
        })
        
        logger.info("Experiment completed!")
        
    def _load_datasets(self) -> Dict[str, Dict[str, List[str]]]:
        """Load different types of datasets."""
        datasets = {}
        
        # Load pretraining data
        texts, labels = self.dataset_loader.load_fineweb_sample(num_samples=200)
        datasets["pretraining"] = {"texts": texts, "labels": labels}
        
        # Load math data
        texts, labels = self.dataset_loader.load_math_qa_data(num_samples=100)
        datasets["math"] = {"texts": texts, "labels": labels}
        
        # Load code data
        texts, labels = self.dataset_loader.load_code_data(num_samples=100)
        datasets["code"] = {"texts": texts, "labels": labels}
        
        # Load capability prompts
        texts, labels = self.prompt_generator.create_balanced_prompt_set(samples_per_category=20)
        datasets["capabilities"] = {"texts": texts, "labels": labels}
        
        # Log dataset statistics
        for name, data in datasets.items():
            logger.info(f"{name}: {len(data['texts'])} samples")
            if self.use_wandb:
                wandb.log({f"dataset_size_{name}": len(data['texts'])})
        
        return datasets
    
    def _collect_router_activations(
        self, 
        datasets: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, torch.Tensor]:
        """Collect router activations for all datasets."""
        all_router_activations = {}
        
        for dataset_name, data in datasets.items():
            logger.info(f"Collecting layer activations for {dataset_name}")
            
            # First collect layer activations
            layer_acts = self.activation_collector.collect_layer_activations(
                data["texts"],
                batch_size=8,
                max_length=512
            )
            
            # Then compute router activations from layer activations
            router_acts = self.activation_collector.collect_router_activations(layer_acts)
            
            # Store with dataset prefix
            for layer_key, activations in router_acts.items():
                full_key = f"{dataset_name}_{layer_key}"
                all_router_activations[full_key] = activations
                
            logger.info(f"Collected router activations for {len(router_acts)} layers")
        
        return all_router_activations
    
    def _analyze_router_weights(self) -> Dict[str, Any]:
        """Analyze router weight matrices."""
        # Extract router weights
        router_weights = self.router_analyzer.extract_all_router_weights()
        
        # Compute similarity matrices
        similarity_matrix = self.router_analyzer.compute_router_similarity_matrix(
            router_weights, similarity_metric="cosine"
        )
        
        # Analyze subspaces
        subspace_analysis = self.router_analyzer.analyze_router_subspaces(
            router_weights, n_components=10
        )
        
        # Compute cross-layer alignment
        alignment_matrix = self.router_analyzer.compute_cross_layer_alignment(
            router_weights, method="principal_angles"
        )
        
        results = {
            "weights": router_weights,
            "similarity_matrix": similarity_matrix,
            "subspace_analysis": subspace_analysis,
            "alignment_matrix": alignment_matrix,
        }
        
        # Log key metrics
        if self.use_wandb:
            if len(similarity_matrix.shape) >= 2:
                mean_similarity = torch.mean(similarity_matrix).item()
                wandb.log({"mean_router_similarity": mean_similarity})
            
            if len(alignment_matrix.shape) >= 2:
                mean_alignment = torch.mean(alignment_matrix).item()
                wandb.log({"mean_cross_layer_alignment": mean_alignment})
        
        return results
    
    def _analyze_routing_patterns(
        self, 
        router_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze routing patterns from activations."""
        routing_patterns = {}
        
        # Group activations by dataset and extract router logits
        datasets = {}
        for key, activations in router_activations.items():
            # Parse key format: dataset_name_layer_X_router_logits/probs
            if "_router_logits" in key:
                parts = key.split("_")
                dataset_name = "_".join(parts[:-3])  # Everything before layer_X_router
                layer_key = f"{parts[-3]}_{parts[-2]}_router"  # layer_X_router
                
                if dataset_name not in datasets:
                    datasets[dataset_name] = {}
                datasets[dataset_name][layer_key] = activations
        
        # Analyze patterns for each dataset
        for dataset_name, dataset_activations in datasets.items():
            logger.info(f"Analyzing routing patterns for {dataset_name}")
            
            patterns = self.router_analyzer.analyze_routing_patterns(
                dataset_activations, temperature=1.0
            )
            routing_patterns[dataset_name] = patterns
            
            # Log key metrics
            if self.use_wandb:
                for layer_key, layer_patterns in patterns.items():
                    wandb.log({
                        f"{dataset_name}_{layer_key}_mean_entropy": layer_patterns["mean_routing_entropy"],
                        f"{dataset_name}_{layer_key}_load_balance": layer_patterns["load_balance_coefficient"],
                        f"{dataset_name}_{layer_key}_concentration": layer_patterns["routing_concentration"],
                    })
        
        return routing_patterns
    
    def _analyze_correlations(
        self, 
        router_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Analyze correlations between layers and datasets."""
        correlations = {}
        
        # Group by dataset and extract router logits
        datasets = {}
        for key, activations in router_activations.items():
            # Parse key format: dataset_name_layer_X_router_logits/probs
            if "_router_logits" in key:
                parts = key.split("_")
                dataset_name = "_".join(parts[:-3])  # Everything before layer_X_router
                layer_name = f"{parts[-3]}_{parts[-2]}"  # layer_X
                
                if dataset_name not in datasets:
                    datasets[dataset_name] = {}
                datasets[dataset_name][layer_name] = activations
        
        # Compute correlations within each dataset
        for dataset_name, dataset_activations in datasets.items():
            logger.info(f"Computing correlations for {dataset_name}")
            
            # Manual correlation computation
            layer_keys = sorted([k for k in dataset_activations.keys() if k.startswith("layer_")])
            if len(layer_keys) > 1:
                # Stack activations: [num_layers, num_samples, num_experts]
                stacked_acts = torch.stack([dataset_activations[key] for key in layer_keys])
                
                # Flatten to [num_layers, num_samples * num_experts]
                flattened_acts = stacked_acts.flatten(start_dim=1)
                
                # Compute Pearson correlation
                corr_matrix = torch.corrcoef(flattened_acts)
                correlations[f"{dataset_name}_correlations"] = corr_matrix
                
                # Log correlation statistics
                if self.use_wandb and corr_matrix.numel() > 0:
                    mean_corr = torch.mean(corr_matrix).item()
                    std_corr = torch.std(corr_matrix).item()
                    wandb.log({
                        f"{dataset_name}_mean_correlation": mean_corr,
                        f"{dataset_name}_std_correlation": std_corr,
                    })
        
        # Compute cross-dataset correlations
        logger.info("Computing cross-dataset correlations")
        # This would compare routing patterns between different data types
        # Implementation depends on specific analysis goals
        
        return correlations
    
    def _generate_visualizations(
        self,
        router_activations: Dict[str, torch.Tensor],
        router_weights: Dict[str, Any],
        routing_patterns: Dict[str, Any],
        correlations: Dict[str, torch.Tensor]
    ):
        """Generate visualizations for the analysis."""
        
        # Plot router weight similarity
        if "similarity_matrix" in router_weights:
            sim_matrix = router_weights["similarity_matrix"]
            if sim_matrix.numel() > 0:
                fig = self.plotter.plot_similarity_matrix(
                    sim_matrix.mean(dim=(2, 3)),  # Average over experts
                    title="Router Weight Similarity Across Layers"
                )
                fig.savefig(self.output_dir / "router_similarity.png")
                if self.use_wandb:
                    wandb.log({"router_similarity_plot": wandb.Image(fig)})
        
        # Plot routing patterns
        for dataset_name, patterns in routing_patterns.items():
            # Extract expert usage across layers
            expert_usage_data = []
            for layer_key, layer_patterns in patterns.items():
                layer_idx = int(layer_key.split("_")[1])
                expert_usage = layer_patterns["expert_usage"].cpu().numpy()
                expert_usage_data.append(expert_usage)
            
            if expert_usage_data:
                expert_usage_matrix = np.stack(expert_usage_data)
                fig = self.plotter.plot_expert_usage_heatmap(
                    expert_usage_matrix,
                    title=f"Expert Usage - {dataset_name.title()}"
                )
                fig.savefig(self.output_dir / f"expert_usage_{dataset_name}.png")
                if self.use_wandb:
                    wandb.log({f"expert_usage_{dataset_name}": wandb.Image(fig)})
        
        # Plot correlations
        for corr_name, corr_matrix in correlations.items():
            if corr_matrix.numel() > 0:
                fig = self.plotter.plot_correlation_matrix(
                    corr_matrix.cpu().numpy(),
                    title=f"Layer Correlations - {corr_name}"
                )
                fig.savefig(self.output_dir / f"correlations_{corr_name}.png")
                if self.use_wandb:
                    wandb.log({f"correlations_{corr_name}": wandb.Image(fig)})
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        # Save router activations
        for key, activations in results["router_activations"].items():
            np.save(self.output_dir / f"activations_{key}.npy", activations.cpu().numpy())
        
        # Save router weights
        router_weights = results["router_weights"]["weights"]
        for layer_key, weights in router_weights.items():
            np.save(self.output_dir / f"router_weights_{layer_key}.npy", weights.cpu().numpy())
        
        # Save analysis results as JSON-serializable format
        import json
        
        # Convert routing patterns to serializable format
        serializable_patterns = {}
        for dataset_name, patterns in results["routing_patterns"].items():
            serializable_patterns[dataset_name] = {}
            for layer_key, layer_patterns in patterns.items():
                serializable_patterns[dataset_name][layer_key] = {
                    k: v.tolist() if torch.is_tensor(v) else v
                    for k, v in layer_patterns.items()
                    if k not in ["top_expert_indices", "top_expert_probs"]  # Skip complex tensors
                }
        
        with open(self.output_dir / "routing_patterns.json", "w") as f:
            json.dump(serializable_patterns, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Run the basic routing analysis experiment."""
    experiment = BasicRoutingAnalysis(
        model_name="allenai/OLMoE-1B-7B-0924",
        output_dir="results/basic_routing",
        use_wandb=True
    )
    
    experiment.run_experiment()


if __name__ == "__main__":
    main()
