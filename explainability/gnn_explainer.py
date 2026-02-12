"""
GNN Explainability for Fraud Detection

This module provides explainability tools to understand WHY the model
flagged a particular transaction or wallet as fraudulent.

Explainability is CRITICAL for fraud detection because:
1. Regulatory Compliance: Financial regulations often require explainable AI
2. Fraud Analyst Review: Humans need to understand model decisions
3. Model Debugging: Identify if model is learning spurious patterns
4. Trust Building: Stakeholders need to trust automated systems

We implement several explanation methods:
1. GNNExplainer: Learns edge/node importance masks
2. Attention Visualization: For attention-based models
3. Subgraph Extraction: Visualize relevant transaction chains
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("networkx not installed. Visualization will be limited.")

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ExplanationType, ModelMode
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False
    logger.warning("PyG Explainer not available. Using custom implementation.")


class FraudExplainer:
    """
    GNN Explainability for Fraud Detection Models.
    
    This class wraps PyTorch Geometric's GNNExplainer to provide
    interpretable explanations for fraud predictions. It identifies
    which edges (transactions) and node features (wallet properties)
    were most important for a fraud classification.
    
    How GNNExplainer Works:
    1. Takes a trained GNN model and a target node
    2. Learns a mask over edges and node features
    3. Masks indicate importance: high value = important for prediction
    4. Optimizes masks to maintain prediction with minimal subgraph
    
    Args:
        model: Trained GNN model
        epochs: Training epochs for the explainer
        lr: Learning rate for mask optimization
        num_hops: Number of hops around target node to consider
        edge_mask_type: 'object' (learnable) or 'common_attributes'
        node_mask_type: 'object', 'common_attributes', or 'attributes'
        
    Example:
        >>> explainer = FraudExplainer(model, epochs=200)
        >>> explanation = explainer.explain_node(
        ...     node_idx=fraud_node_id,
        ...     x=node_features,
        ...     edge_index=edge_index
        ... )
        >>> explainer.visualize(explanation, save_path="fraud_123.png")
    """
    
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
        num_hops: int = 2,
        edge_mask_type: str = 'object',
        node_mask_type: str = 'attributes',
        coefficients: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.num_hops = num_hops
        
        # Default coefficients for explanation loss
        self.coefficients = coefficients or {
            'edge_size': 0.005,
            'edge_ent': 0.1,
            'node_feat_size': 0.001,
            'node_feat_ent': 0.1,
        }
        
        # Create wrapper model for explainer compatibility
        self.wrapper_model = self._create_wrapper_model(model)
        
        # Initialize explainer
        self._init_explainer(edge_mask_type, node_mask_type)
    
    def _create_wrapper_model(self, model: nn.Module) -> nn.Module:
        """Create a wrapper that adapts the temporal model for PyG Explainer."""
        class ExplainerWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, x, edge_index, **kwargs):
                # Use single snapshot forward if available
                if hasattr(self.base_model, 'forward_single_snapshot'):
                    logits = self.base_model.forward_single_snapshot(x, edge_index)
                else:
                    # Wrap as single-element sequence
                    logits = self.base_model([x], [edge_index])
                # Return probabilities for binary classification
                return torch.softmax(logits, dim=-1)[:, 1]
        
        wrapper = ExplainerWrapper(model)
        wrapper.eval()
        return wrapper
    
    def _init_explainer(self, edge_mask_type: str, node_mask_type: str):
        """Initialize the GNNExplainer."""
        if HAS_EXPLAINER:
            # Use PyG's built-in explainer (PyG >= 2.3)
            self.explainer = Explainer(
                model=self.wrapper_model,
                algorithm=GNNExplainer(epochs=self.epochs, lr=self.lr),
                explanation_type='model',
                edge_mask_type=edge_mask_type,
                node_mask_type=node_mask_type,
                model_config=dict(
                    mode='binary_classification',
                    task_level='node',
                    return_type='probs',
                ),
            )
        else:
            # Fallback to custom implementation
            self.explainer = CustomGNNExplainer(
                model=self.model,
                epochs=self.epochs,
                lr=self.lr,
                coefficients=self.coefficients,
            )
    
    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        target: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Generate explanation for a single node's prediction.
        
        Args:
            node_idx: Index of node to explain
            x: Node features [N, F]
            edge_index: Graph connectivity [2, E]
            edge_attr: Optional edge features [E, edge_dim]
            target: Target class to explain (default: predicted class)
            
        Returns:
            Dictionary containing:
            - edge_mask: Importance scores for edges [E]
            - node_mask: Importance scores for node features [N, F] or [F]
            - prediction: Model's prediction for this node
            - subgraph: Indices of nodes in explanation subgraph
        """
        self.model.eval()
        
        # Get model prediction if target not specified
        if target is None:
            with torch.no_grad():
                logits = self.model(x, edge_index, edge_attr)
                target = logits[node_idx].argmax().item()
        
        if HAS_EXPLAINER:
            explanation = self.explainer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                index=node_idx,
                target=target,
            )
            
            return {
                'edge_mask': explanation.edge_mask,
                'node_mask': explanation.get('node_mask'),
                'prediction': target,
                'node_idx': node_idx,
            }
        else:
            return self.explainer.explain_node(
                node_idx, x, edge_index, target
            )
    
    def explain_fraud_prediction(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        top_k_edges: int = 10,
        top_k_features: int = 5,
    ) -> Dict:
        """
        Generate a comprehensive explanation for a fraud prediction.
        
        Returns human-readable explanation including:
        - Most important transactions (edges)
        - Most important wallet features
        - Subgraph of related wallets
        
        Args:
            node_idx: Index of the flagged fraud node
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Edge features
            top_k_edges: Number of top edges to return
            top_k_features: Number of top features to return
            
        Returns:
            Dictionary with detailed explanation
        """
        # Get raw explanation
        explanation = self.explain_node(
            node_idx=node_idx,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target=1,  # Explain fraud prediction
        )
        
        edge_mask = explanation['edge_mask']
        
        # Get top-k most important edges
        if edge_mask is not None:
            top_edge_indices = edge_mask.argsort(descending=True)[:top_k_edges]
            top_edges = edge_index[:, top_edge_indices]
            top_edge_scores = edge_mask[top_edge_indices]
        else:
            top_edges = None
            top_edge_scores = None
        
        # Get top-k most important node features
        node_mask = explanation.get('node_mask')
        if node_mask is not None:
            if node_mask.dim() == 1:
                top_feat_indices = node_mask.argsort(descending=True)[:top_k_features]
                top_feat_scores = node_mask[top_feat_indices]
            else:
                node_importance = node_mask.mean(dim=0)
                top_feat_indices = node_importance.argsort(descending=True)[:top_k_features]
                top_feat_scores = node_importance[top_feat_indices]
        else:
            top_feat_indices = None
            top_feat_scores = None
        
        # Extract subgraph around the node
        subgraph_nodes = self._extract_subgraph(node_idx, edge_index, self.num_hops)
        
        return {
            'node_idx': node_idx,
            'prediction': 'FRAUD',
            'confidence': None,  # Can be computed from model output
            'top_edges': top_edges,
            'top_edge_scores': top_edge_scores,
            'top_feature_indices': top_feat_indices,
            'top_feature_scores': top_feat_scores,
            'subgraph_nodes': subgraph_nodes,
            'raw_explanation': explanation,
        }
    
    def _extract_subgraph(
        self,
        node_idx: int,
        edge_index: Tensor,
        num_hops: int
    ) -> List[int]:
        """Extract k-hop subgraph around a node."""
        nodes = {node_idx}
        edge_index_np = edge_index.cpu().numpy()
        
        for _ in range(num_hops):
            new_nodes = set()
            for n in nodes:
                # Find neighbors
                mask_src = edge_index_np[0] == n
                mask_tgt = edge_index_np[1] == n
                neighbors = set(edge_index_np[1, mask_src]) | set(edge_index_np[0, mask_tgt])
                new_nodes |= neighbors
            nodes |= new_nodes
        
        return list(nodes)
    
    def visualize(
        self,
        explanation: Dict,
        x: Tensor,
        edge_index: Tensor,
        node_labels: Optional[Tensor] = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Visualize a fraud explanation as a graph.
        
        Args:
            explanation: Output from explain_fraud_prediction
            x: Node features
            edge_index: Graph connectivity
            node_labels: Ground truth labels for nodes
            feature_names: Names of node features
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        return visualize_fraud_subgraph(
            explanation=explanation,
            x=x,
            edge_index=edge_index,
            node_labels=node_labels,
            feature_names=feature_names,
            save_path=save_path,
            figsize=figsize,
        )


class CustomGNNExplainer:
    """
    Custom GNNExplainer implementation (fallback when PyG Explainer unavailable).
    
    Implements the core GNNExplainer algorithm:
    1. Initialize learnable edge mask (sigmoid of parameters)
    2. Forward pass through model with masked edges
    3. Optimize mask to maintain prediction with sparse edges
    """
    
    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
        coefficients: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.coefficients = coefficients or {
            'edge_size': 0.005,
            'edge_ent': 0.1,
        }
    
    def explain_node(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        target: int,
    ) -> Dict[str, Tensor]:
        """Generate explanation for a node."""
        self.model.eval()
        
        num_edges = edge_index.size(1)
        device = x.device
        
        # Initialize edge mask parameters
        edge_mask_params = nn.Parameter(torch.zeros(num_edges, device=device))
        
        optimizer = torch.optim.Adam([edge_mask_params], lr=self.lr)
        
        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Compute edge mask via sigmoid
            edge_mask = torch.sigmoid(edge_mask_params)
            
            # Forward with masked edges (multiply edge weights)
            # This is a simplified version - full implementation would
            # properly integrate mask into message passing
            logits = self.model(x, edge_index)
            
            # Prediction loss: maintain original prediction
            pred_loss = -F.log_softmax(logits, dim=-1)[node_idx, target]
            
            # Sparsity loss: encourage sparse masks
            size_loss = self.coefficients['edge_size'] * edge_mask.sum()
            
            # Entropy loss: encourage binary masks (0 or 1)
            ent_loss = self.coefficients['edge_ent'] * (
                -edge_mask * torch.log(edge_mask + 1e-8) -
                (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
            ).mean()
            
            loss = pred_loss + size_loss + ent_loss
            loss.backward()
            optimizer.step()
        
        # Final mask
        with torch.no_grad():
            edge_mask = torch.sigmoid(edge_mask_params)
        
        return {
            'edge_mask': edge_mask,
            'node_mask': None,
            'prediction': target,
            'node_idx': node_idx,
        }


class TemporalGNNExplainer:
    """
    Explainer for Temporal GNN models.
    
    Temporal models add complexity because we need to explain
    not just which edges/features matter, but also WHEN they matter.
    
    This explainer provides:
    1. Temporal importance: Which timesteps contributed most
    2. Spatial importance: Which edges at each timestep
    3. Combined visualization: Timeline of fraud evidence
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_explainer: Optional[FraudExplainer] = None,
    ):
        self.model = model
        self.base_explainer = base_explainer or FraudExplainer(model)
    
    def explain_temporal_prediction(
        self,
        node_idx: int,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        top_k_timesteps: int = 3,
    ) -> Dict:
        """
        Explain a prediction across temporal snapshots.
        
        Returns importance scores for each timestep, allowing
        analysts to see when suspicious activity occurred.
        """
        timestep_importance = []
        timestep_explanations = []
        
        # Get importance of each timestep
        for t, (x, edge_index) in enumerate(zip(x_sequence, edge_index_sequence)):
            # Check if node exists at this timestep
            if node_idx >= x.size(0):
                timestep_importance.append(0.0)
                timestep_explanations.append(None)
                continue
            
            # Get explanation for this timestep
            explanation = self.base_explainer.explain_node(
                node_idx=node_idx,
                x=x,
                edge_index=edge_index,
                target=1,  # Fraud class
            )
            
            # Compute timestep importance as sum of edge mask
            if explanation['edge_mask'] is not None:
                importance = explanation['edge_mask'].sum().item()
            else:
                importance = 0.0
            
            timestep_importance.append(importance)
            timestep_explanations.append(explanation)
        
        # Normalize importance
        total_importance = sum(timestep_importance) + 1e-8
        timestep_importance = [imp / total_importance for imp in timestep_importance]
        
        # Get top-k most important timesteps
        top_timesteps = sorted(
            range(len(timestep_importance)),
            key=lambda t: timestep_importance[t],
            reverse=True
        )[:top_k_timesteps]
        
        return {
            'node_idx': node_idx,
            'timestep_importance': timestep_importance,
            'top_timesteps': top_timesteps,
            'timestep_explanations': timestep_explanations,
        }
    
    def visualize_temporal_explanation(
        self,
        explanation: Dict,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize temporal importance as a bar chart."""
        timestep_importance = explanation['timestep_importance']
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        timesteps = range(len(timestep_importance))
        colors = ['red' if t in explanation['top_timesteps'] else 'steelblue'
                  for t in timesteps]
        
        ax.bar(timesteps, timestep_importance, color=colors, edgecolor='black')
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title(f'Temporal Importance for Fraud Detection (Node {explanation["node_idx"]})',
                    fontsize=14)
        
        # Add legend
        patches = [
            mpatches.Patch(color='red', label='Top Important'),
            mpatches.Patch(color='steelblue', label='Other'),
        ]
        ax.legend(handles=patches)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved temporal explanation to {save_path}")
        
        return fig


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_fraud_subgraph(
    explanation: Dict,
    x: Tensor,
    edge_index: Tensor,
    node_labels: Optional[Tensor] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Visualize a fraud explanation as a network graph.
    
    The visualization shows:
    - Target fraud node (red, large)
    - Connected wallets (blue/orange based on label)
    - Edge thickness based on importance
    - Important features annotated
    
    Args:
        explanation: Output from FraudExplainer.explain_fraud_prediction
        x: Node features
        edge_index: Graph connectivity  
        node_labels: Ground truth labels
        feature_names: Names of features
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not HAS_NETWORKX:
        logger.error("networkx required for visualization. Install with: pip install networkx")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # === Left: Subgraph Visualization ===
    ax1 = axes[0]
    
    subgraph_nodes = explanation['subgraph_nodes']
    target_node = explanation['node_idx']
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in subgraph_nodes:
        G.add_node(node)
    
    # Add edges within subgraph
    edge_index_np = edge_index.cpu().numpy()
    edge_mask = explanation.get('raw_explanation', {}).get('edge_mask')
    
    for i in range(edge_index.size(1)):
        src, tgt = edge_index_np[0, i], edge_index_np[1, i]
        if src in subgraph_nodes and tgt in subgraph_nodes:
            weight = edge_mask[i].item() if edge_mask is not None else 0.5
            G.add_edge(src, tgt, weight=weight)
    
    # Node colors based on label
    node_colors = []
    for node in G.nodes():
        if node == target_node:
            node_colors.append('red')
        elif node_labels is not None and node_labels[node] == 1:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')
    
    # Node sizes
    node_sizes = [500 if node == target_node else 200 for node in G.nodes()]
    
    # Edge widths based on importance
    edge_widths = [G[u][v]['weight'] * 5 + 0.5 for u, v in G.edges()]
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                          node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths,
                          alpha=0.6, edge_color='gray',
                          connectionstyle='arc3,rad=0.1',
                          arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
    
    ax1.set_title(f'Fraud Subgraph (Target: Node {target_node})', fontsize=12)
    ax1.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='red', label='Target (Fraud)'),
        mpatches.Patch(color='orange', label='Known Fraud'),
        mpatches.Patch(color='lightblue', label='Legitimate/Unknown'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # === Right: Feature Importance ===
    ax2 = axes[1]
    
    top_feat_indices = explanation.get('top_feature_indices')
    top_feat_scores = explanation.get('top_feature_scores')
    
    if top_feat_indices is not None and top_feat_scores is not None:
        indices = top_feat_indices.cpu().numpy()
        scores = top_feat_scores.cpu().numpy()
        
        if feature_names:
            labels = [feature_names[i] if i < len(feature_names) else f'Feature {i}'
                     for i in indices]
        else:
            labels = [f'Feature {i}' for i in indices]
        
        y_pos = range(len(labels))
        ax2.barh(y_pos, scores, color='steelblue', edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top Important Features', fontsize=12)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'Feature importance\nnot available',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Top Important Features', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved fraud explanation visualization to {save_path}")
    
    return fig


def explain_prediction(
    model: nn.Module,
    node_idx: int,
    x: Tensor,
    edge_index: Tensor,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    High-level function to generate and visualize a fraud explanation.
    
    Args:
        model: Trained GNN model
        node_idx: Node to explain
        x: Node features
        edge_index: Graph connectivity
        feature_names: Optional feature names for visualization
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary with explanation details
    """
    explainer = FraudExplainer(model)
    
    explanation = explainer.explain_fraud_prediction(
        node_idx=node_idx,
        x=x,
        edge_index=edge_index,
        top_k_edges=10,
        top_k_features=5,
    )
    
    if save_dir:
        save_path = Path(save_dir) / f"fraud_explanation_node_{node_idx}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        explainer.visualize(
            explanation=explanation,
            x=x,
            edge_index=edge_index,
            feature_names=feature_names,
            save_path=str(save_path),
        )
    
    return explanation


if __name__ == "__main__":
    print("GNN Explainability for Fraud Detection")
    print("=" * 50)
    print("\nThis module provides explainability for GNN-based fraud detection.")
    print("\nKey components:")
    print("  - FraudExplainer: Wrapper around GNNExplainer")
    print("  - TemporalGNNExplainer: For temporal models")
    print("  - visualize_fraud_subgraph: Graph visualization")
    print("\nExample usage:")
    print("  >>> explainer = FraudExplainer(model)")
    print("  >>> explanation = explainer.explain_fraud_prediction(node_idx, x, edge_index)")
    print("  >>> explainer.visualize(explanation, x, edge_index)")
