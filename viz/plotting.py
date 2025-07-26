"""
Plotting utilities for MoE routing analysis visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RouterPlotter:
    """
    Plotting utilities for router analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize router plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        title: str = "Router Similarity Matrix",
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot router similarity matrix as heatmap.
        
        Args:
            similarity_matrix: Similarity matrix [num_layers, num_layers]
            title: Plot title
            layer_names: Optional layer names for axes
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(similarity_matrix.shape[0])]
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(layer_names)))
        ax.set_yticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_yticklabels(layer_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score')
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer')
        
        # Add text annotations
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_expert_usage_heatmap(
        self,
        expert_usage: np.ndarray,
        title: str = "Expert Usage Across Layers",
        layer_names: Optional[List[str]] = None,
        expert_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot expert usage as heatmap.
        
        Args:
            expert_usage: Expert usage matrix [num_layers, num_experts]
            title: Plot title
            layer_names: Optional layer names
            expert_names: Optional expert names
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(expert_usage.shape[0])]
        if expert_names is None:
            expert_names = [f"Expert {i}" for i in range(expert_usage.shape[1])]
        
        # Create heatmap
        sns.heatmap(
            expert_usage,
            xticklabels=expert_names,
            yticklabels=layer_names,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Usage Probability'}
        )
        
        ax.set_title(title)
        ax.set_xlabel('Expert')
        ax.set_ylabel('Layer')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_routing_entropy(
        self,
        entropy_data: Dict[str, List[float]],
        title: str = "Routing Entropy Across Layers",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot routing entropy across layers for different datasets.
        
        Args:
            entropy_data: Dictionary mapping dataset names to entropy lists
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for dataset_name, entropies in entropy_data.items():
            layers = list(range(len(entropies)))
            ax.plot(layers, entropies, marker='o', label=dataset_name, linewidth=2)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Routing Entropy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_load_balance(
        self,
        load_balance_data: Dict[str, List[float]],
        title: str = "Load Balance Across Layers",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot load balance coefficients across layers.
        
        Args:
            load_balance_data: Dictionary mapping dataset names to load balance lists
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for dataset_name, load_balances in load_balance_data.items():
            layers = list(range(len(load_balances)))
            ax.plot(layers, load_balances, marker='s', label=dataset_name, linewidth=2)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Load Balance Coefficient')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_interactive_routing_patterns(
        self,
        routing_data: Dict[str, np.ndarray],
        title: str = "Interactive Routing Patterns"
    ) -> go.Figure:
        """
        Create interactive routing pattern visualization using Plotly.
        
        Args:
            routing_data: Dictionary mapping dataset names to routing matrices
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create subplots
        datasets = list(routing_data.keys())
        fig = make_subplots(
            rows=1, cols=len(datasets),
            subplot_titles=datasets,
            shared_yaxes=True
        )
        
        for i, (dataset_name, routing_matrix) in enumerate(routing_data.items()):
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=routing_matrix,
                    colorscale='Viridis',
                    showscale=(i == len(datasets) - 1),  # Only show colorbar for last subplot
                    hovertemplate=f'Layer: %{{y}}<br>Expert: %{{x}}<br>Usage: %{{z:.3f}}<extra></extra>'
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500,
            showlegend=False
        )
        
        # Update axes
        for i in range(len(datasets)):
            fig.update_xaxes(title_text="Expert", row=1, col=i+1)
            if i == 0:
                fig.update_yaxes(title_text="Layer", row=1, col=i+1)
        
        return fig


class ExpertPlotter:
    """
    Plotting utilities for expert analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize expert plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_expert_similarity_matrix(
        self,
        similarity_matrices: Dict[str, np.ndarray],
        title: str = "Expert Similarity Within Layers",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot expert similarity matrices for multiple layers.
        
        Args:
            similarity_matrices: Dictionary mapping layer names to similarity matrices
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_layers = len(similarity_matrices)
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (layer_name, sim_matrix) in enumerate(similarity_matrices.items()):
            ax = axes[i] if num_layers > 1 else axes[0]
            
            im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f"{layer_name}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Expert")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_layers, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_expert_specialization(
        self,
        specialization_data: Dict[str, Dict[str, float]],
        title: str = "Expert Specialization Scores",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot expert specialization scores.
        
        Args:
            specialization_data: Nested dict {layer: {expert: score}}
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame for easier plotting
        data_rows = []
        for layer_name, experts in specialization_data.items():
            for expert_name, score in experts.items():
                data_rows.append({
                    'Layer': layer_name,
                    'Expert': expert_name,
                    'Specialization Score': score
                })
        
        df = pd.DataFrame(data_rows)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create grouped bar plot
        sns.barplot(data=df, x='Layer', y='Specialization Score', hue='Expert', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Specialization Score')
        ax.legend(title='Expert', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_weight_norms(
        self,
        weight_norms: Dict[str, Dict[str, float]],
        title: str = "Expert Weight Norms",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot expert weight norms across layers.
        
        Args:
            weight_norms: Nested dict {layer: {expert: norm}}
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        layers = list(weight_norms.keys())
        experts = list(next(iter(weight_norms.values())).keys())
        
        x = np.arange(len(layers))
        width = 0.8 / len(experts)
        
        for i, expert in enumerate(experts):
            norms = [weight_norms[layer][expert] for layer in layers]
            ax.bar(x + i * width, norms, width, label=expert)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Weight Norm')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(experts) - 1) / 2)
        ax.set_xticklabels(layers)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class CorrelationPlotter:
    """
    Plotting utilities for correlation analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize correlation plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        title: str = "Layer Correlation Matrix",
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix between layers.
        
        Args:
            correlation_matrix: Correlation matrix [num_layers, num_layers]
            title: Plot title
            layer_names: Optional layer names
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(correlation_matrix.shape[0])]
        
        # Create correlation heatmap
        sns.heatmap(
            correlation_matrix,
            xticklabels=layer_names,
            yticklabels=layer_names,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Correlation'}
        )
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_cross_dataset_correlations(
        self,
        correlation_data: Dict[str, np.ndarray],
        title: str = "Cross-Dataset Correlations",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlations between different datasets.
        
        Args:
            correlation_data: Dictionary mapping dataset pairs to correlation values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Convert to DataFrame for easier plotting
        data_rows = []
        for dataset_pair, correlations in correlation_data.items():
            for i, corr in enumerate(correlations):
                data_rows.append({
                    'Dataset Pair': dataset_pair,
                    'Layer': i,
                    'Correlation': corr
                })
        
        df = pd.DataFrame(data_rows)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create line plot
        sns.lineplot(data=df, x='Layer', y='Correlation', hue='Dataset Pair', 
                    marker='o', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Correlation')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_evolution(
        self,
        correlation_evolution: Dict[str, List[np.ndarray]],
        title: str = "Correlation Evolution During Training",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot how correlations evolve during training (if checkpoint data available).
        
        Args:
            correlation_evolution: Dict mapping checkpoints to correlation matrices
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        checkpoints = list(correlation_evolution.keys())
        num_checkpoints = len(checkpoints)
        
        cols = min(4, num_checkpoints)
        rows = (num_checkpoints + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if num_checkpoints == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (checkpoint, corr_matrix) in enumerate(correlation_evolution.items()):
            ax = axes[i] if num_checkpoints > 1 else axes[0]
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f"Checkpoint {checkpoint}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Layer")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_checkpoints, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

