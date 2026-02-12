#!/usr/bin/env python3
"""
Ablation Study for Fraud Detection Model

Systematically evaluates the contribution of each improvement component:
1. Feature enhancement
2. TemporalGAT vs GConvGRU architecture  
3. Focal loss vs standard cross-entropy
4. Class weighting
5. Threshold optimization
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_curve, 
    f1_score, 
    precision_score, 
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import load_elliptic_dataset, create_temporal_splits, TemporalGraphSnapshot
from models.temporal_gnn import create_model

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
        x=enhanced_x,
        edge_index=snapshot.edge_index,
        edge_attr=snapshot.edge_attr,
        y=snapshot.y,
        train_mask=snapshot.train_mask,
        timestep=snapshot.timestep,
        node_ids=snapshot.node_ids,
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.92, gamma=3.5, pos_weight=25.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device)
        )
        
        pos_weight = torch.where(
            targets == 1,
            torch.tensor(self.pos_weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        loss = alpha_weight * pos_weight * focal_weight * ce_loss
        return loss.mean()


class WeightedCrossEntropy(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.where(
            targets == 1,
            torch.tensor(self.pos_weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        return (weights * ce_loss).mean()


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, min_precision: float = 0.35) -> Tuple[float, Dict]:
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


def train_and_evaluate(
    train_seq: List[TemporalGraphSnapshot],
    val_seq: List[TemporalGraphSnapshot],
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    architecture: str = 'temporal_gat',
    use_enhanced_features: bool = True,
    use_focal_loss: bool = True,
    pos_weight: float = 25.0,
    use_threshold_opt: bool = True,
    hidden_channels: List[int] = [256, 128, 64],
    epochs: int = 150,
    seed: int = 42,
) -> Dict:
    """Train a model with specified configuration and evaluate."""
    
    set_seed(seed)
    
    # Feature enhancement
    if use_enhanced_features:
        train_seq = [compute_enhanced_features(s) for s in train_seq]
        val_seq = [compute_enhanced_features(s) for s in val_seq]
        test_seq = [compute_enhanced_features(s) for s in test_seq]
    
    node_input_dim = train_seq[0].x.size(1)
    
    # Create model
    model = create_model(
        architecture=architecture,
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=hidden_channels,
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.92, gamma=3.5, pos_weight=pos_weight)
    else:
        criterion = WeightedCrossEntropy(pos_weight=pos_weight)
    
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
    
    # Metrics
    pr_auc = average_precision_score(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    if use_threshold_opt:
        threshold, metrics = find_best_threshold(y_true, y_probs, min_precision=0.35)
    else:
        threshold = 0.5
        y_pred = (y_probs >= threshold).astype(int)
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
    
    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs per config')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", "ablation_study")
    
    logger.info("=" * 60)
    logger.info("ABLATION STUDY")
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
    
    # Define ablation configurations
    configs = [
        {
            'name': 'Full Model (Best)',
            'architecture': 'temporal_gat',
            'use_enhanced_features': True,
            'use_focal_loss': True,
            'pos_weight': 25.0,
            'use_threshold_opt': True,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'w/o Enhanced Features',
            'architecture': 'temporal_gat',
            'use_enhanced_features': False,
            'use_focal_loss': True,
            'pos_weight': 25.0,
            'use_threshold_opt': True,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'GConvGRU (instead of GAT)',
            'architecture': 'gconv_gru',
            'use_enhanced_features': True,
            'use_focal_loss': True,
            'pos_weight': 25.0,
            'use_threshold_opt': True,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'w/o Focal Loss',
            'architecture': 'temporal_gat',
            'use_enhanced_features': True,
            'use_focal_loss': False,
            'pos_weight': 25.0,
            'use_threshold_opt': True,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'w/o Class Weighting',
            'architecture': 'temporal_gat',
            'use_enhanced_features': True,
            'use_focal_loss': True,
            'pos_weight': 1.0,
            'use_threshold_opt': True,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'w/o Threshold Optimization',
            'architecture': 'temporal_gat',
            'use_enhanced_features': True,
            'use_focal_loss': True,
            'pos_weight': 25.0,
            'use_threshold_opt': False,
            'hidden_channels': [256, 128, 64],
        },
        {
            'name': 'Small Model (128,64)',
            'architecture': 'temporal_gat',
            'use_enhanced_features': True,
            'use_focal_loss': True,
            'pos_weight': 25.0,
            'use_threshold_opt': True,
            'hidden_channels': [128, 64],
        },
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Config {i+1}/{len(configs)}: {config['name']}")
        logger.info(f"{'=' * 50}")
        
        try:
            def copy_node_ids(node_ids):
                if node_ids is None:
                    return None
                if isinstance(node_ids, torch.Tensor):
                    return node_ids.clone()
                return node_ids.copy() if hasattr(node_ids, 'copy') else node_ids
            
            # Make copies of sequences (they get modified by feature enhancement)
            train_copy = [TemporalGraphSnapshot(
                x=s.x.clone(), edge_index=s.edge_index.clone(),
                edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
                y=s.y.clone(), train_mask=s.train_mask.clone(),
                timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
            ) for s in train_seq]
            
            val_copy = [TemporalGraphSnapshot(
                x=s.x.clone(), edge_index=s.edge_index.clone(),
                edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
                y=s.y.clone(), train_mask=s.train_mask.clone(),
                timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
            ) for s in val_seq]
            
            test_copy = [TemporalGraphSnapshot(
                x=s.x.clone(), edge_index=s.edge_index.clone(),
                edge_attr=s.edge_attr.clone() if s.edge_attr is not None else None,
                y=s.y.clone(), train_mask=s.train_mask.clone(),
                timestep=s.timestep, node_ids=copy_node_ids(s.node_ids)
            ) for s in test_seq]
            
            metrics = train_and_evaluate(
                train_seq=train_copy,
                val_seq=val_copy,
                test_seq=test_copy,
                device=device,
                architecture=config['architecture'],
                use_enhanced_features=config['use_enhanced_features'],
                use_focal_loss=config['use_focal_loss'],
                pos_weight=config['pos_weight'],
                use_threshold_opt=config['use_threshold_opt'],
                hidden_channels=config['hidden_channels'],
                epochs=args.epochs,
                seed=args.seed,
            )
            
            result = {'config': config['name'], **metrics}
            results.append(result)
            
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"  F1:     {metrics['f1']:.4f}")
            logger.info(f"  Prec:   {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            
        except Exception as e:
            logger.error(f"Error with config {config['name']}: {e}")
            results.append({'config': config['name'], 'error': str(e)})
    
    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Configuration':<35} {'PR-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 80)
    
    for r in results:
        if 'error' not in r:
            logger.info(f"{r['config']:<35} {r['pr_auc']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
        else:
            logger.info(f"{r['config']:<35} {'ERROR':>10}")
    
    # Save results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_results = [r for r in results if 'error' not in r]
    names = [r['config'] for r in valid_results]
    pr_aucs = [r['pr_auc'] for r in valid_results]
    f1s = [r['f1'] for r in valid_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pr_aucs, width, label='PR-AUC', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
