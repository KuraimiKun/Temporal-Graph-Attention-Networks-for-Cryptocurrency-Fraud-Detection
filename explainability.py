#!/usr/bin/env python3
"""
GNN Explainability Analysis for Fraud Detection

Uses GNNExplainer to understand which features and graph structures
contribute most to fraud predictions.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.manifold import TSNE

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import load_elliptic_dataset, create_temporal_splits, TemporalGraphSnapshot
from models.temporal_gnn import create_model

logger = logging.getLogger(__name__)


# Feature names for the Elliptic dataset
FEATURE_NAMES = [
    # Local features (94 total - first 94)
    *[f"local_feat_{i}" for i in range(94)],
    # Aggregated features (72 total)
    *[f"agg_feat_{i}" for i in range(72)],
    # Enhanced features (4)
    "in_degree", "out_degree", "io_ratio", "neighbor_count"
]


def compute_enhanced_features(snapshot: TemporalGraphSnapshot) -> TemporalGraphSnapshot:
    """Add graph structural features."""
    x = snapshot.x
    edge_index = snapshot.edge_index
    num_nodes = x.size(0)
    
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        in_degree = torch.zeros(num_nodes, device=x.device)
        out_degree = torch.zeros(num_nodes, device=x.device)
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        in_degree = in_degree / (in_degree.max() + 1e-6)
        out_degree = out_degree / (out_degree.max() + 1e-6)
        io_ratio = in_degree / (out_degree + 1e-6)
        io_ratio = torch.clamp(io_ratio, 0, 10) / 10
        neighbor_count = in_degree + out_degree
    else:
        in_degree = torch.zeros(num_nodes, device=x.device)
        out_degree = torch.zeros(num_nodes, device=x.device)
        io_ratio = torch.zeros(num_nodes, device=x.device)
        neighbor_count = torch.zeros(num_nodes, device=x.device)
    
    new_features = torch.stack([in_degree, out_degree, io_ratio, neighbor_count], dim=1)
    enhanced_x = torch.cat([x, new_features], dim=1)
    
    return TemporalGraphSnapshot(
        x=enhanced_x,
        edge_index=snapshot.edge_index,
        edge_attr=snapshot.edge_attr,
        y=snapshot.y,
        train_mask=snapshot.train_mask,
        timestep=snapshot.timestep,
        node_ids=snapshot.node_ids,
    )


class TemporalModelWrapper(torch.nn.Module):
    """Wrapper to make temporal model compatible with GNNExplainer."""
    
    def __init__(self, model, x_sequence, edge_sequence):
        super().__init__()
        self.model = model
        self.x_sequence = x_sequence  # All but last timestep
        self.edge_sequence = edge_sequence
    
    def forward(self, x, edge_index):
        """Forward with last timestep's x and edge_index."""
        full_x = self.x_sequence + [x]
        full_edge = self.edge_sequence + [edge_index]
        return self.model(full_x, full_edge)


def analyze_feature_importance(
    model: torch.nn.Module,
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    num_samples: int = 100,
    output_dir: Path = None,
) -> Dict:
    """
    Analyze feature importance using gradient-based attribution.
    """
    logger.info("Analyzing feature importance...")
    
    model.eval()
    test_seq_device = [s.to(device) for s in test_seq]
    
    x_seq = [s.x for s in test_seq_device]
    edge_seq = [s.edge_index for s in test_seq_device]
    
    last_snapshot = test_seq_device[-1]
    mask = last_snapshot.train_mask
    labels = last_snapshot.y[mask]
    
    # Get fraud and legit indices
    fraud_indices = torch.where((labels == 1))[0]
    legit_indices = torch.where((labels == 0))[0]
    
    # Sample nodes
    num_fraud = min(num_samples, len(fraud_indices))
    num_legit = min(num_samples, len(legit_indices))
    
    fraud_sample = fraud_indices[torch.randperm(len(fraud_indices))[:num_fraud]]
    legit_sample = legit_indices[torch.randperm(len(legit_indices))[:num_legit]]
    
    # Compute gradients for feature importance
    fraud_importance = []
    legit_importance = []
    
    for idx in fraud_sample:
        # Enable gradients on input
        x_last = x_seq[-1].clone().requires_grad_(True)
        x_input = x_seq[:-1] + [x_last]
        
        out = model(x_input, edge_seq)
        fraud_prob = F.softmax(out, dim=1)[mask][idx, 1]
        fraud_prob.backward()
        
        grad = x_last.grad[mask][idx].abs().cpu().numpy()
        fraud_importance.append(grad)
    
    for idx in legit_sample:
        x_last = x_seq[-1].clone().requires_grad_(True)
        x_input = x_seq[:-1] + [x_last]
        
        out = model(x_input, edge_seq)
        legit_prob = F.softmax(out, dim=1)[mask][idx, 0]
        legit_prob.backward()
        
        grad = x_last.grad[mask][idx].abs().cpu().numpy()
        legit_importance.append(grad)
    
    fraud_importance = np.mean(fraud_importance, axis=0)
    legit_importance = np.mean(legit_importance, axis=0)
    
    # Get top features
    num_features = len(fraud_importance)
    feature_names = FEATURE_NAMES[:num_features] if num_features <= len(FEATURE_NAMES) else [f"feat_{i}" for i in range(num_features)]
    
    fraud_top_idx = np.argsort(fraud_importance)[-20:][::-1]
    legit_top_idx = np.argsort(legit_importance)[-20:][::-1]
    
    results = {
        'fraud_top_features': [(feature_names[i], float(fraud_importance[i])) for i in fraud_top_idx],
        'legit_top_features': [(feature_names[i], float(legit_importance[i])) for i in legit_top_idx],
        'fraud_importance': fraud_importance.tolist(),
        'legit_importance': legit_importance.tolist(),
    }
    
    # Plot feature importance
    if output_dir:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Fraud features
        ax = axes[0]
        top_fraud_names = [feature_names[i] for i in fraud_top_idx[:15]]
        top_fraud_vals = [fraud_importance[i] for i in fraud_top_idx[:15]]
        ax.barh(range(15), top_fraud_vals[::-1], color='red', alpha=0.7)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_fraud_names[::-1])
        ax.set_xlabel('Importance (Gradient Magnitude)')
        ax.set_title('Top 15 Features for Fraud Detection')
        
        # Legit features
        ax = axes[1]
        top_legit_names = [feature_names[i] for i in legit_top_idx[:15]]
        top_legit_vals = [legit_importance[i] for i in legit_top_idx[:15]]
        ax.barh(range(15), top_legit_vals[::-1], color='green', alpha=0.7)
        ax.set_yticks(range(15))
        ax.set_yticklabels(top_legit_names[::-1])
        ax.set_xlabel('Importance (Gradient Magnitude)')
        ax.set_title('Top 15 Features for Legitimate Classification')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot to {output_dir / 'feature_importance.png'}")
    
    return results


def generate_tsne_embeddings(
    model: torch.nn.Module,
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    output_dir: Path = None,
) -> Dict:
    """Generate t-SNE visualization of learned embeddings."""
    logger.info("Generating t-SNE embeddings...")
    
    model.eval()
    test_seq_device = [s.to(device) for s in test_seq]
    
    x_seq = [s.x for s in test_seq_device]
    edge_seq = [s.edge_index for s in test_seq_device]
    
    last_snapshot = test_seq_device[-1]
    mask = last_snapshot.train_mask
    labels = last_snapshot.y[mask].cpu().numpy()
    
    # Get embeddings before final classification layer
    with torch.no_grad():
        # Get intermediate representation
        out = model(x_seq, edge_seq)
        probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        
        # Use logits as embeddings (or you can hook into intermediate layers)
        embeddings = out[mask].cpu().numpy()
    
    # Apply t-SNE (use max_iter for sklearn >= 1.5)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    results = {
        'embeddings_2d': embeddings_2d.tolist(),
        'labels': labels.tolist(),
        'probs': probs.tolist(),
    }
    
    if output_dir:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # By true label
        ax = axes[0]
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='RdYlGn_r', alpha=0.6, s=20
        )
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Embeddings by True Label')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fraud (1) / Legit (0)')
        
        # By predicted probability
        ax = axes[1]
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=probs, cmap='RdYlGn_r', alpha=0.6, s=20
        )
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE Embeddings by Predicted Fraud Probability')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fraud Probability')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE plot to {output_dir / 'tsne_embeddings.png'}")
    
    return results


def analyze_attention_patterns(
    model: torch.nn.Module,
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    output_dir: Path = None,
) -> Dict:
    """Analyze attention patterns in GAT layers (if applicable)."""
    logger.info("Analyzing attention patterns...")
    
    model.eval()
    test_seq_device = [s.to(device) for s in test_seq]
    
    last_snapshot = test_seq_device[-1]
    mask = last_snapshot.train_mask
    labels = last_snapshot.y[mask].cpu().numpy()
    
    # Get edge attention weights if available
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'alpha') or (isinstance(output, tuple) and len(output) > 1):
            try:
                alpha = output[1] if isinstance(output, tuple) else output.alpha
                attention_weights.append(alpha.detach().cpu())
            except:
                pass
    
    # Register hooks on GAT layers
    hooks = []
    for name, module in model.named_modules():
        if 'gat' in name.lower() or 'attention' in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))
    
    x_seq = [s.x for s in test_seq_device]
    edge_seq = [s.edge_index for s in test_seq_device]
    
    with torch.no_grad():
        out = model(x_seq, edge_seq)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    results = {
        'num_attention_layers': len(attention_weights),
        'attention_stats': [],
    }
    
    for i, attn in enumerate(attention_weights):
        if attn is not None:
            stats = {
                'layer': i,
                'mean': float(attn.mean()),
                'std': float(attn.std()),
                'max': float(attn.max()),
                'min': float(attn.min()),
            }
            results['attention_stats'].append(stats)
    
    logger.info(f"Found {len(attention_weights)} attention layers")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GNN Explainability Analysis")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model checkpoint')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for feature importance')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", "explainability")
    
    logger.info("=" * 60)
    logger.info("GNN EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_elliptic_dataset()
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset,
        train_timesteps=42,
        val_timesteps=5,
        test_timesteps=2,
    )
    
    # Enhance features
    test_seq = [compute_enhanced_features(s) for s in test_seq]
    node_input_dim = test_seq[0].x.size(1)
    
    # Create model (same architecture as best model)
    logger.info("\nCreating model...")
    model = create_model(
        architecture='temporal_gat',
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=[256, 128, 64],
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    # Load weights if available
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.warning("No model path provided, using untrained model for demonstration")
    
    # Run analyses
    logger.info("\n" + "-" * 40)
    logger.info("1. Feature Importance Analysis")
    logger.info("-" * 40)
    feature_results = analyze_feature_importance(
        model, test_seq, device,
        num_samples=args.num_samples,
        output_dir=output_dir
    )
    
    logger.info("\nTop 10 features for FRAUD detection:")
    for name, importance in feature_results['fraud_top_features'][:10]:
        logger.info(f"  {name}: {importance:.4f}")
    
    logger.info("\n" + "-" * 40)
    logger.info("2. t-SNE Embedding Visualization")
    logger.info("-" * 40)
    tsne_results = generate_tsne_embeddings(
        model, test_seq, device,
        output_dir=output_dir
    )
    
    logger.info("\n" + "-" * 40)
    logger.info("3. Attention Pattern Analysis")
    logger.info("-" * 40)
    attention_results = analyze_attention_patterns(
        model, test_seq, device,
        output_dir=output_dir
    )
    
    # Save all results
    all_results = {
        'feature_importance': feature_results,
        'tsne': {
            'num_samples': len(tsne_results['labels']),
            'num_fraud': sum(tsne_results['labels']),
        },
        'attention': attention_results,
    }
    
    with open(output_dir / 'explainability_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
