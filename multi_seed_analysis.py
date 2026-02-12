#!/usr/bin/env python3
"""
Multi-Seed Statistical Analysis

Runs multiple training runs with different random seeds to:
1. Get mean ± std for all metrics
2. Assess statistical significance
3. Generate confidence intervals
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import load_elliptic_dataset, create_temporal_splits, TemporalGraphSnapshot
from models.temporal_gnn import create_model
from train_improved import ImprovedFocalLoss

logger = logging.getLogger(__name__)


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


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, min_precision: float = 0.35):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    valid_mask = precisions[:-1] >= min_precision
    
    if valid_mask.sum() == 0:
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    else:
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(recalls[:-1][valid_indices])]
        best_threshold = thresholds[best_idx]
    
    y_pred = (y_scores >= best_threshold).astype(int)
    
    return best_threshold, {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def train_single_run(
    seed: int,
    train_seq_orig: List[TemporalGraphSnapshot],
    val_seq_orig: List[TemporalGraphSnapshot],
    test_seq_orig: List[TemporalGraphSnapshot],
    device: torch.device,
    epochs: int = 150,
    architecture: str = 'temporal_gat',
    hidden_channels: List[int] = [256, 128, 64],
    pos_weight: float = 25.0,
) -> Dict:
    """Train a single model run with given seed."""
    
    set_seed(seed)
    
    def copy_node_ids(node_ids):
        if node_ids is None:
            return None
        if isinstance(node_ids, torch.Tensor):
            return node_ids.clone()
        return node_ids.copy() if hasattr(node_ids, 'copy') else node_ids
    
    # Make copies and enhance features
    train_seq = [compute_enhanced_features(TemporalGraphSnapshot(
        x=s.x.clone(), edge_index=s.edge_index.clone(),
        edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
        y=s.y.clone(), train_mask=s.train_mask.clone(),
        timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
    )) for s in train_seq_orig]
    
    val_seq = [compute_enhanced_features(TemporalGraphSnapshot(
        x=s.x.clone(), edge_index=s.edge_index.clone(),
        edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
        y=s.y.clone(), train_mask=s.train_mask.clone(),
        timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
    )) for s in val_seq_orig]
    
    test_seq = [compute_enhanced_features(TemporalGraphSnapshot(
        x=s.x.clone(), edge_index=s.edge_index.clone(),
        edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
        y=s.y.clone(), train_mask=s.train_mask.clone(),
        timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
    )) for s in test_seq_orig]
    
    node_input_dim = train_seq[0].x.size(1)
    
    model = create_model(
        architecture=architecture,
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=hidden_channels,
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    criterion = ImprovedFocalLoss(alpha=0.92, gamma=3.5, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_pr_auc = 0.0
    best_model_state = None
    no_improve = 0
    patience = 30
    
    for epoch in range(epochs):
        model.train()
        train_seq_device = [s.to(device) for s in train_seq]
        
        for seq_start in range(0, len(train_seq_device) - 4, 3):
            seq = train_seq_device[seq_start:seq_start + 5]
            if len(seq) < 5:
                continue
            
            x_seq = [s.x for s in seq]
            edge_seq = [s.edge_index for s in seq]
            
            optimizer.zero_grad()
            out = model(x_seq, edge_seq)
            
            last_snapshot = seq[-1]
            mask = last_snapshot.train_mask
            y = last_snapshot.y[mask]
            pred = out[mask]
            
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_seq_device = [s.to(device) for s in val_seq]
        x_val = [s.x for s in val_seq_device]
        edge_val = [s.edge_index for s in val_seq_device]
        
        with torch.no_grad():
            out = model(x_val, edge_val)
            last_val = val_seq_device[-1]
            mask = last_val.train_mask
            y = last_val.y[mask]
            probs = F.softmax(out[mask], dim=1)[:, 1]
            
            if y.sum() > 0:
                pr_auc = average_precision_score(y.cpu().numpy(), probs.cpu().numpy())
            else:
                pr_auc = 0.0
        
        scheduler.step(pr_auc)
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
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
    
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    threshold, metrics = find_best_threshold(y_true, y_probs, min_precision=0.35)
    
    return {
        'seed': seed,
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'threshold': float(threshold),
    }


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute mean, std, and confidence intervals."""
    
    metrics = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall']
    stats_dict = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        
        # 95% confidence interval
        n = len(values)
        se = std / np.sqrt(n)
        t_val = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se
        
        stats_dict[metric] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'values': [float(v) for v in values],
        }
    
    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Statistical Analysis")
    parser.add_argument('--num-seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--architecture', type=str, default='temporal_gat')
    
    args = parser.parse_args()
    
    setup_logging()
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", f"multi_seed_{args.num_seeds}runs")
    
    logger.info("=" * 60)
    logger.info("MULTI-SEED STATISTICAL ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Number of runs: {args.num_seeds}")
    logger.info(f"Architecture: {args.architecture}")
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_elliptic_dataset()
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset, train_timesteps=42, val_timesteps=5, test_timesteps=2,
    )
    
    # Run multiple seeds
    seeds = [42, 123, 456, 789, 1234, 2024, 999, 314, 271, 161][:args.num_seeds]
    results = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Run {i+1}/{args.num_seeds} (seed={seed})")
        logger.info(f"{'=' * 40}")
        
        result = train_single_run(
            seed=seed,
            train_seq_orig=train_seq,
            val_seq_orig=val_seq,
            test_seq_orig=test_seq,
            device=device,
            epochs=args.epochs,
            architecture=args.architecture,
        )
        results.append(result)
        
        logger.info(f"  PR-AUC: {result['pr_auc']:.4f}")
        logger.info(f"  F1:     {result['f1']:.4f}")
        logger.info(f"  Prec:   {result['precision']:.4f}")
        logger.info(f"  Recall: {result['recall']:.4f}")
    
    # Compute statistics
    stats_dict = compute_statistics(results)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'95% CI':>20}")
    logger.info("-" * 80)
    
    for metric, data in stats_dict.items():
        ci_str = f"[{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]"
        logger.info(f"{metric:<15} {data['mean']:>10.4f} {data['std']:>10.4f} {data['min']:>10.4f} {data['max']:>10.4f} {ci_str:>20}")
    
    # LaTeX table format
    logger.info("\n" + "=" * 60)
    logger.info("LATEX TABLE FORMAT")
    logger.info("=" * 60)
    logger.info("\\begin{table}[h]")
    logger.info("\\centering")
    logger.info("\\begin{tabular}{lcc}")
    logger.info("\\toprule")
    logger.info("Metric & Mean $\\pm$ Std & 95\\% CI \\\\")
    logger.info("\\midrule")
    
    metric_names = {
        'pr_auc': 'PR-AUC',
        'roc_auc': 'ROC-AUC', 
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
    }
    
    for metric, data in stats_dict.items():
        name = metric_names.get(metric, metric)
        logger.info(f"{name} & {data['mean']:.4f} $\\pm$ {data['std']:.4f} & [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}] \\\\")
    
    logger.info("\\bottomrule")
    logger.info("\\end{tabular}")
    logger.info(f"\\caption{{Performance metrics across {args.num_seeds} runs with different random seeds.}}")
    logger.info("\\label{tab:multi-seed}")
    logger.info("\\end{table}")
    
    # Save results
    output = {
        'config': {
            'num_seeds': args.num_seeds,
            'seeds': seeds,
            'architecture': args.architecture,
            'epochs': args.epochs,
        },
        'individual_runs': results,
        'statistics': stats_dict,
    }
    
    with open(output_dir / 'multi_seed_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax = axes[0]
    metrics_to_plot = ['pr_auc', 'f1', 'precision', 'recall']
    data_to_plot = [stats_dict[m]['values'] for m in metrics_to_plot]
    labels = ['PR-AUC', 'F1', 'Precision', 'Recall']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['steelblue', 'coral', 'green', 'purple']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Metric Distribution Across {args.num_seeds} Seeds')
    ax.grid(axis='y', alpha=0.3)
    
    # Bar plot with error bars
    ax = axes[1]
    x = np.arange(len(metrics_to_plot))
    means = [stats_dict[m]['mean'] for m in metrics_to_plot]
    stds = [stats_dict[m]['std'] for m in metrics_to_plot]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('Mean ± Std Across Seeds')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_seed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
