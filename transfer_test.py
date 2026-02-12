#!/usr/bin/env python3
"""
Transfer Learning Test: Train on Elliptic → Test on Elliptic++

This script:
1. Loads a model trained on the original Elliptic dataset
2. Tests it on the Elliptic++ dataset (which has additional features)
3. Reports transfer performance metrics

The goal is to demonstrate generalization capability of the trained model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import TemporalGraphSnapshot
from models.temporal_gnn import create_model

logger = logging.getLogger(__name__)


def load_elliptic_plus_plus(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Elliptic++ dataset."""
    features_df = pd.read_csv(data_dir / "txs_features.csv")
    classes_df = pd.read_csv(data_dir / "txs_classes.csv")
    edges_df = pd.read_csv(data_dir / "txs_edgelist.csv")
    
    return features_df, classes_df, edges_df


def create_temporal_snapshots_elliptic_pp(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    num_original_features: int = 165,  # Match original Elliptic feature count (94 local + 72 aggregated - 1 for 0-indexing in original)
) -> List[TemporalGraphSnapshot]:
    """Create temporal graph snapshots from Elliptic++ data."""
    
    # Merge features with classes
    data = features_df.merge(classes_df, on='txId', how='left')
    
    # Map classes: 1=illicit, 2=licit, 3=unknown
    # Convert to binary: 1=illicit (fraud), 0=licit, -1=unknown
    data['label'] = data['class'].map({1: 1, 2: 0, 3: -1})
    
    snapshots = []
    timesteps = sorted(data['Time step'].unique())
    
    # Create node ID mapping for edges
    all_tx_ids = data['txId'].values
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}
    
    for t in timesteps:
        # Get nodes at this timestep
        timestep_data = data[data['Time step'] == t].copy()
        node_ids = timestep_data['txId'].values
        
        # Create local node mapping for this timestep
        local_tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(node_ids)}
        
        # Get features (use only first 166 to match original Elliptic)
        # Skip txId and Time step columns
        feature_cols = [c for c in features_df.columns if c not in ['txId', 'Time step']]
        
        # Use first num_original_features to match trained model
        if len(feature_cols) > num_original_features:
            feature_cols = feature_cols[:num_original_features]
        
        x = torch.tensor(timestep_data[feature_cols].values, dtype=torch.float32)
        
        # Pad if fewer features
        if x.size(1) < num_original_features:
            padding = torch.zeros(x.size(0), num_original_features - x.size(1))
            x = torch.cat([x, padding], dim=1)
        
        # Get labels
        y = torch.tensor(timestep_data['label'].values, dtype=torch.long)
        
        # Create mask for labeled nodes (exclude unknown class=-1)
        train_mask = y >= 0
        
        # Get edges for this timestep
        timestep_edges = edges_df[
            edges_df['txId1'].isin(node_ids) & edges_df['txId2'].isin(node_ids)
        ]
        
        if len(timestep_edges) > 0:
            src = timestep_edges['txId1'].map(local_tx_to_idx).values
            dst = timestep_edges['txId2'].map(local_tx_to_idx).values
            
            # Filter out any NaN mappings
            valid = ~(np.isnan(src) | np.isnan(dst))
            src = src[valid].astype(int)
            dst = dst[valid].astype(int)
            
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        snapshot = TemporalGraphSnapshot(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            y=y,
            train_mask=train_mask,
            timestep=t,
            node_ids=torch.tensor(node_ids),
        )
        snapshots.append(snapshot)
    
    return snapshots


def compute_enhanced_features(snapshot: TemporalGraphSnapshot) -> TemporalGraphSnapshot:
    """Add graph structural features (same as training)."""
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


def evaluate_transfer(
    model: torch.nn.Module,
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on transfer dataset."""
    
    model.eval()
    test_seq_device = [s.to(device) for s in test_seq]
    x_test = [s.x for s in test_seq_device]
    edge_test = [s.edge_index for s in test_seq_device]
    
    with torch.no_grad():
        out = model(x_test, edge_test)
        
        # Get predictions for last timestep
        last_test = test_seq_device[-1]
        mask = last_test.train_mask
        
        y_true = last_test.y[mask].cpu().numpy()
        y_probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        y_pred = (y_probs >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'pr_auc': average_precision_score(y_true, y_probs),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'threshold': threshold,
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    })
    
    return metrics


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, Dict]:
    """Find threshold that maximizes F1."""
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_probs >= best_threshold).astype(int)
    
    return best_threshold, {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Test")
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--elliptic-pp-dir', type=str, 
                       default='data/Elliptic++ Dataset/raw',
                       help='Path to Elliptic++ dataset')
    parser.add_argument('--test-timesteps', type=int, default=5,
                       help='Number of timesteps to use for testing')
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", "transfer_test")
    
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING TEST")
    logger.info("Train: Elliptic (Bitcoin) → Test: Elliptic++ (Extended)")
    logger.info("=" * 60)
    
    # Load Elliptic++ dataset
    logger.info("\nLoading Elliptic++ dataset...")
    elliptic_pp_dir = Path(args.elliptic_pp_dir)
    features_df, classes_df, edges_df = load_elliptic_plus_plus(elliptic_pp_dir)
    
    logger.info(f"  Transactions: {len(features_df)}")
    logger.info(f"  Features: {features_df.shape[1] - 2}")  # Exclude txId, Time step
    logger.info(f"  Edges: {len(edges_df)}")
    
    # Class distribution
    class_counts = classes_df['class'].value_counts()
    logger.info(f"  Classes: 1(illicit)={class_counts.get(1, 0)}, 2(licit)={class_counts.get(2, 0)}, 3(unknown)={class_counts.get(3, 0)}")
    
    # Create temporal snapshots
    logger.info("\nCreating temporal snapshots...")
    all_snapshots = create_temporal_snapshots_elliptic_pp(
        features_df, classes_df, edges_df, 
        num_original_features=165  # Match original Elliptic (94 local + 72 aggregated - 1)
    )
    
    # Use last N timesteps for testing
    test_seq = all_snapshots[-args.test_timesteps:]
    logger.info(f"  Using last {args.test_timesteps} timesteps for transfer test")
    
    # Enhance features (same preprocessing as training)
    test_seq = [compute_enhanced_features(s) for s in test_seq]
    
    node_input_dim = test_seq[0].x.size(1)
    logger.info(f"  Input features: {node_input_dim}")
    
    # Count labeled samples in test set
    last_snapshot = test_seq[-1]
    n_labeled = last_snapshot.train_mask.sum().item()
    n_fraud = (last_snapshot.y[last_snapshot.train_mask] == 1).sum().item()
    n_licit = (last_snapshot.y[last_snapshot.train_mask] == 0).sum().item()
    logger.info(f"  Test samples: {n_labeled} (fraud={n_fraud}, licit={n_licit})")
    
    # Load trained model
    logger.info(f"\nLoading model from {args.model_path}...")
    
    model = create_model(
        architecture='temporal_gat',
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=[256, 128, 64],
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info("  Model loaded successfully")
    
    # Evaluate with different thresholds
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFER TEST RESULTS")
    logger.info("=" * 60)
    
    # Get predictions for threshold optimization
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
    
    # Test with default threshold
    logger.info("\n--- Default Threshold (0.5) ---")
    metrics_default = evaluate_transfer(model, test_seq, device, threshold=0.5)
    logger.info(f"  PR-AUC:    {metrics_default['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics_default['roc_auc']:.4f}")
    logger.info(f"  F1:        {metrics_default['f1']:.4f}")
    logger.info(f"  Precision: {metrics_default['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_default['recall']:.4f}")
    logger.info(f"  Confusion: TN={metrics_default['tn']}, FP={metrics_default['fp']}, FN={metrics_default['fn']}, TP={metrics_default['tp']}")
    
    # Test with optimized threshold
    best_threshold, _ = find_best_threshold(y_true, y_probs)
    logger.info(f"\n--- Optimized Threshold ({best_threshold:.4f}) ---")
    metrics_optimized = evaluate_transfer(model, test_seq, device, threshold=best_threshold)
    logger.info(f"  PR-AUC:    {metrics_optimized['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics_optimized['roc_auc']:.4f}")
    logger.info(f"  F1:        {metrics_optimized['f1']:.4f}")
    logger.info(f"  Precision: {metrics_optimized['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_optimized['recall']:.4f}")
    logger.info(f"  Confusion: TN={metrics_optimized['tn']}, FP={metrics_optimized['fp']}, FN={metrics_optimized['fn']}, TP={metrics_optimized['tp']}")
    
    # Test with training threshold (0.614)
    logger.info(f"\n--- Training Threshold (0.614) ---")
    metrics_train_thresh = evaluate_transfer(model, test_seq, device, threshold=0.614)
    logger.info(f"  PR-AUC:    {metrics_train_thresh['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics_train_thresh['roc_auc']:.4f}")
    logger.info(f"  F1:        {metrics_train_thresh['f1']:.4f}")
    logger.info(f"  Precision: {metrics_train_thresh['precision']:.4f}")
    logger.info(f"  Recall:    {metrics_train_thresh['recall']:.4f}")
    logger.info(f"  Confusion: TN={metrics_train_thresh['tn']}, FP={metrics_train_thresh['fp']}, FN={metrics_train_thresh['fn']}, TP={metrics_train_thresh['tp']}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFER LEARNING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<15} {'Default(0.5)':>12} {'Optimized':>12} {'Train(0.614)':>12}")
    logger.info("-" * 55)
    logger.info(f"{'PR-AUC':<15} {metrics_default['pr_auc']:>12.4f} {metrics_optimized['pr_auc']:>12.4f} {metrics_train_thresh['pr_auc']:>12.4f}")
    logger.info(f"{'ROC-AUC':<15} {metrics_default['roc_auc']:>12.4f} {metrics_optimized['roc_auc']:>12.4f} {metrics_train_thresh['roc_auc']:>12.4f}")
    logger.info(f"{'F1':<15} {metrics_default['f1']:>12.4f} {metrics_optimized['f1']:>12.4f} {metrics_train_thresh['f1']:>12.4f}")
    logger.info(f"{'Precision':<15} {metrics_default['precision']:>12.4f} {metrics_optimized['precision']:>12.4f} {metrics_train_thresh['precision']:>12.4f}")
    logger.info(f"{'Recall':<15} {metrics_default['recall']:>12.4f} {metrics_optimized['recall']:>12.4f} {metrics_train_thresh['recall']:>12.4f}")
    
    # Save results
    import json
    
    # Helper to convert numpy/torch types to native Python types
    def to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        return obj
    
    results = {
        'model_path': str(args.model_path),
        'elliptic_pp_dir': str(args.elliptic_pp_dir),
        'test_timesteps': args.test_timesteps,
        'n_test_samples': n_labeled,
        'n_fraud': n_fraud,
        'n_licit': n_licit,
        'metrics_default': to_python(metrics_default),
        'metrics_optimized': to_python(metrics_optimized),
        'metrics_train_thresh': to_python(metrics_train_thresh),
        'best_threshold': float(best_threshold),
    }
    
    with open(output_dir / 'transfer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)
    
    if metrics_optimized['pr_auc'] > 0.3:
        logger.info("✓ POSITIVE TRANSFER: Model shows meaningful generalization")
        logger.info(f"  - PR-AUC {metrics_optimized['pr_auc']:.4f} indicates learned fraud patterns transfer")
    else:
        logger.info("✗ LIMITED TRANSFER: Model struggles on new data")
        logger.info("  - May need domain adaptation or fine-tuning")
    
    if metrics_optimized['recall'] > 0.5:
        logger.info(f"✓ Good fraud detection capability (Recall={metrics_optimized['recall']:.2%})")
    else:
        logger.info(f"! Lower recall on transfer ({metrics_optimized['recall']:.2%}) - expected for zero-shot")


if __name__ == "__main__":
    main()
