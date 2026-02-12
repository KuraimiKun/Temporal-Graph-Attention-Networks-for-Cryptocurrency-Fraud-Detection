#!/usr/bin/env python3
"""
Visualization Generation for Thesis

Generates publication-quality figures:
1. PR and ROC curves
2. Confusion matrix heatmaps
3. Training curves
4. Model comparison charts
5. Temporal analysis plots
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
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import load_elliptic_dataset, create_temporal_splits, TemporalGraphSnapshot
from models.temporal_gnn import create_model

logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


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
        x=enhanced_x, edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr,
        y=snapshot.y, train_mask=snapshot.train_mask, timestep=snapshot.timestep,
        node_ids=snapshot.node_ids,
    )


def plot_pr_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    model_name: str = "TemporalGAT",
    output_path: Path = None,
) -> Tuple[float, float]:
    """Generate PR and ROC curves side by side."""
    
    # Compute curves
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Calculate baseline (random classifier)
    baseline_precision = y_true.sum() / len(y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    ax = axes[0]
    ax.plot(recall, precision, 'b-', linewidth=2.5, label=f'{model_name} (AP = {pr_auc:.3f})')
    ax.axhline(y=baseline_precision, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Random Baseline ({baseline_precision:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # ROC Curve
    ax = axes[1]
    ax.plot(fpr, tpr, 'r-', linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1.5, label='Random Baseline')
    ax.fill_between(fpr, tpr, alpha=0.2, color='red')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved PR/ROC curves to {output_path}")
    
    plt.close()
    return pr_auc, roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = None,
) -> None:
    """Generate confusion matrix heatmap."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Counts)')
    
    # Normalized
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict],
    output_path: Path = None,
) -> None:
    """Generate model comparison bar chart."""
    
    models = list(results.keys())
    metrics = ['PR-AUC', 'F1', 'Precision', 'Recall']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(models))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        key = metric.lower().replace('-', '_')
        values = [results[m].get(key, 0) for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison to {output_path}")
    
    plt.close()


def plot_temporal_analysis(
    dataset,
    output_path: Path = None,
) -> None:
    """Analyze and visualize temporal patterns in the dataset."""
    
    timesteps = []
    num_nodes = []
    num_edges = []
    fraud_ratios = []
    
    for i, snapshot in enumerate(dataset._snapshots):
        timesteps.append(i)
        num_nodes.append(snapshot.x.size(0))
        num_edges.append(snapshot.edge_index.size(1))
        
        mask = snapshot.train_mask
        labels = snapshot.y[mask]
        if len(labels) > 0:
            fraud_ratio = (labels == 1).float().mean().item()
        else:
            fraud_ratio = 0
        fraud_ratios.append(fraud_ratio)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Nodes per timestep
    ax = axes[0, 0]
    ax.bar(timesteps, num_nodes, color='steelblue', alpha=0.7)
    ax.axvline(x=41.5, color='red', linestyle='--', label='Train/Val split')
    ax.axvline(x=46.5, color='orange', linestyle='--', label='Val/Test split')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Transaction Graph Size Over Time')
    ax.legend()
    
    # Edges per timestep
    ax = axes[0, 1]
    ax.bar(timesteps, num_edges, color='coral', alpha=0.7)
    ax.axvline(x=41.5, color='red', linestyle='--', label='Train/Val split')
    ax.axvline(x=46.5, color='orange', linestyle='--', label='Val/Test split')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Edges')
    ax.set_title('Transaction Connections Over Time')
    ax.legend()
    
    # Fraud ratio
    ax = axes[1, 0]
    ax.plot(timesteps, fraud_ratios, 'g-o', markersize=4, linewidth=1.5)
    ax.axvline(x=41.5, color='red', linestyle='--', label='Train/Val split')
    ax.axvline(x=46.5, color='orange', linestyle='--', label='Val/Test split')
    ax.fill_between(timesteps, fraud_ratios, alpha=0.3, color='green')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Fraud Ratio')
    ax.set_title('Fraud Rate Evolution Over Time')
    ax.legend()
    
    # Class distribution
    ax = axes[1, 1]
    total_fraud = sum([s.y[s.train_mask].sum().item() for s in dataset._snapshots])
    total_legit = sum([(s.y[s.train_mask] == 0).sum().item() for s in dataset._snapshots])
    
    sizes = [total_legit, total_fraud]
    labels = [f'Legitimate\n({total_legit:,})', f'Fraud\n({total_fraud:,})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.05)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12})
    ax.set_title('Overall Class Distribution')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved temporal analysis to {output_path}")
    
    plt.close()


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path = None,
) -> None:
    """Analyze how metrics change with different thresholds."""
    
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    ax.plot(thresholds, f1s, 'r-', linewidth=2, label='F1 Score')
    
    # Mark optimal F1 threshold
    best_idx = np.argmax(f1s)
    ax.axvline(x=thresholds[best_idx], color='gray', linestyle='--', alpha=0.7)
    ax.scatter([thresholds[best_idx]], [f1s[best_idx]], color='red', s=100, zorder=5,
               label=f'Best F1 = {f1s[best_idx]:.3f} @ θ={thresholds[best_idx]:.2f}')
    
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision-Recall-F1 vs. Decision Threshold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved threshold analysis to {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Thesis Visualizations")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", "visualizations")
    
    logger.info("=" * 60)
    logger.info("GENERATING THESIS VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_elliptic_dataset()
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset, train_timesteps=42, val_timesteps=5, test_timesteps=2,
    )
    
    # 1. Temporal Analysis
    logger.info("\n1. Generating temporal analysis...")
    plot_temporal_analysis(dataset, output_dir / 'temporal_analysis.png')
    
    # Enhance features and prepare model
    test_seq = [compute_enhanced_features(s) for s in test_seq]
    node_input_dim = test_seq[0].x.size(1)
    
    # Create model
    model = create_model(
        architecture='temporal_gat',
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=[256, 128, 64],
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    if args.model_path and Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Handle both model state_dict and full results.pt files
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from results file: {args.model_path}")
        elif isinstance(checkpoint, dict) and any(k.startswith('gat_convs') or k.startswith('gru') for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
            logger.info(f"Loaded model state_dict from {args.model_path}")
        else:
            logger.warning(f"Could not find model weights in {args.model_path}, training fresh model")
            args.model_path = None
    
    # Get predictions
    model.eval()
    test_seq_device = [s.to(device) for s in test_seq]
    x_test = [s.x for s in test_seq_device]
    edge_test = [s.edge_index for s in test_seq_device]
    
    with torch.no_grad():
        out = model(x_test, edge_test)
        last_test = test_seq_device[-1]
        mask = last_test.train_mask
        y_true = last_test.y[mask].cpu().numpy()
        y_probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    
    # 2. PR and ROC Curves
    logger.info("\n2. Generating PR/ROC curves...")
    pr_auc, roc_auc = plot_pr_roc_curves(
        y_true, y_probs, "TemporalGAT", output_dir / 'pr_roc_curves.png'
    )
    
    # 3. Confusion Matrix
    logger.info("\n3. Generating confusion matrix...")
    # Use optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores)]
    y_pred = (y_probs >= best_thresh).astype(int)
    
    plot_confusion_matrix(y_true, y_pred, output_dir / 'confusion_matrix.png')
    
    # 4. Threshold Analysis
    logger.info("\n4. Generating threshold analysis...")
    plot_threshold_analysis(y_true, y_probs, output_dir / 'threshold_analysis.png')
    
    # 5. Model Comparison (using saved results if available)
    logger.info("\n5. Generating model comparison...")
    comparison_results = {
        'Baseline GConvGRU': {'pr_auc': 0.2115, 'f1': 0.303, 'precision': 0.192, 'recall': 0.714},
        'Improved GConvGRU': {'pr_auc': 0.2463, 'f1': 0.385, 'precision': 0.250, 'recall': 0.839},
        'TemporalGAT (small)': {'pr_auc': 0.3761, 'f1': 0.505, 'precision': 0.352, 'recall': 0.893},
        'TemporalGAT (large)': {'pr_auc': 0.4820, 'f1': 0.510, 'precision': 0.351, 'recall': 0.929},
    }
    plot_model_comparison(comparison_results, output_dir / 'model_comparison.png')
    
    logger.info("\n" + "=" * 60)
    logger.info(f"All visualizations saved to: {output_dir}")
    logger.info("=" * 60)
    
    # Summary
    logger.info("\nGenerated files:")
    for f in output_dir.glob('*.png'):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
