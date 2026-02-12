#!/usr/bin/env python3
"""
Baseline Comparisons for Fraud Detection

Compares the Temporal GNN against non-GNN baselines:
1. Random Forest
2. XGBoost
3. Logistic Regression
4. Multi-Layer Perceptron (MLP)
5. Simple GNN (no temporal modeling)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import set_seed, get_device, setup_logging, create_experiment_dir
from data.elliptic_loader import load_elliptic_dataset, create_temporal_splits, TemporalGraphSnapshot
from models.temporal_gnn import create_model

logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Skipping XGBoost baseline.")


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


def prepare_tabular_data(
    train_seq: List[TemporalGraphSnapshot],
    val_seq: List[TemporalGraphSnapshot],
    test_seq: List[TemporalGraphSnapshot],
    use_enhanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert graph data to tabular format for traditional ML."""
    
    if use_enhanced:
        train_seq = [compute_enhanced_features(s) for s in train_seq]
        val_seq = [compute_enhanced_features(s) for s in val_seq]
        test_seq = [compute_enhanced_features(s) for s in test_seq]
    
    # Collect all training data
    X_train_list = []
    y_train_list = []
    
    for snapshot in train_seq:
        mask = snapshot.train_mask
        X_train_list.append(snapshot.x[mask].numpy())
        y_train_list.append(snapshot.y[mask].numpy())
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Validation data (use last snapshot)
    last_val = val_seq[-1]
    mask = last_val.train_mask
    X_val = last_val.x[mask].numpy()
    y_val = last_val.y[mask].numpy()
    
    # Test data (use last snapshot)
    last_test = test_seq[-1]
    mask = last_test.train_mask
    X_test = last_test.x[mask].numpy()
    y_test = last_test.y[mask].numpy()
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_sklearn_model(model, X_test, y_test, model_name: str) -> Dict:
    """Evaluate sklearn model and return metrics."""
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)
        y_probs = 1 / (1 + np.exp(-y_probs))  # Convert to probability
    
    # Metrics
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    # Find optimal threshold
    threshold, metrics = find_best_threshold(y_test, y_probs, min_precision=0.35)
    
    return {
        'model': model_name,
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'threshold': float(threshold),
    }


def train_random_forest(X_train, y_train, X_val, y_val) -> RandomForestClassifier:
    """Train Random Forest with class balancing."""
    logger.info("Training Random Forest...")
    
    # Calculate class weights
    n_fraud = (y_train == 1).sum()
    n_legit = (y_train == 0).sum()
    class_weight = {0: 1.0, 1: n_legit / n_fraud}
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with class balancing."""
    if not HAS_XGBOOST:
        return None
    
    logger.info("Training XGBoost...")
    
    # Calculate scale_pos_weight
    n_fraud = (y_train == 1).sum()
    n_legit = (y_train == 0).sum()
    scale_pos_weight = n_legit / n_fraud
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def train_logistic_regression(X_train, y_train, scaler) -> LogisticRegression:
    """Train Logistic Regression."""
    logger.info("Training Logistic Regression...")
    
    n_fraud = (y_train == 1).sum()
    n_legit = (y_train == 0).sum()
    class_weight = {0: 1.0, 1: n_legit / n_fraud}
    
    X_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    
    return model


def train_mlp(X_train, y_train, X_val, y_val, scaler) -> MLPClassifier:
    """Train MLP classifier."""
    logger.info("Training MLP...")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Oversample minority class
    n_fraud = (y_train == 1).sum()
    n_legit = (y_train == 0).sum()
    oversample_ratio = min(5, n_legit // n_fraud)
    
    fraud_indices = np.where(y_train == 1)[0]
    oversampled_indices = np.repeat(fraud_indices, oversample_ratio)
    all_indices = np.concatenate([np.arange(len(y_train)), oversampled_indices])
    
    X_train_balanced = X_train_scaled[all_indices]
    y_train_balanced = y_train[all_indices]
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train_balanced, y_train_balanced)
    
    return model


def train_temporal_gat(
    train_seq: List[TemporalGraphSnapshot],
    val_seq: List[TemporalGraphSnapshot],
    test_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    epochs: int = 150,
) -> Dict:
    """Train TemporalGAT model (our proposed method)."""
    from train_improved import ImprovedFocalLoss
    
    logger.info("Training TemporalGAT (our method)...")
    
    # Enhance features
    train_seq = [compute_enhanced_features(s) for s in train_seq]
    val_seq = [compute_enhanced_features(s) for s in val_seq]
    test_seq = [compute_enhanced_features(s) for s in test_seq]
    
    node_input_dim = train_seq[0].x.size(1)
    
    model = create_model(
        architecture='temporal_gat',
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=[256, 128, 64],
        num_classes=2,
        dropout=0.4,
    ).to(device)
    
    criterion = ImprovedFocalLoss(alpha=0.92, gamma=3.5, pos_weight=25.0)
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
        'model': 'TemporalGAT (Ours)',
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'f1': float(metrics['f1']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'threshold': float(threshold),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparisons")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-gnn', action='store_true', help='Skip GNN training (faster)')
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(args.seed)
    device = get_device()
    
    output_dir = create_experiment_dir("outputs", "baseline_comparison")
    
    logger.info("=" * 60)
    logger.info("BASELINE COMPARISON STUDY")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("\nLoading dataset...")
    dataset = load_elliptic_dataset()
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset, train_timesteps=42, val_timesteps=5, test_timesteps=2,
    )
    
    # Prepare tabular data for traditional ML
    logger.info("\nPreparing tabular data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_tabular_data(
        train_seq, val_seq, test_seq, use_enhanced=True
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Fraud ratio (train): {y_train.mean():.2%}")
    
    results = []
    scaler = StandardScaler()
    
    # 1. Logistic Regression
    logger.info("\n" + "-" * 40)
    lr_model = train_logistic_regression(X_train, y_train, StandardScaler())
    X_test_scaled = StandardScaler().fit(X_train).transform(X_test)
    lr_results = evaluate_sklearn_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    results.append(lr_results)
    logger.info(f"LR: PR-AUC={lr_results['pr_auc']:.4f}, F1={lr_results['f1']:.4f}")
    
    # 2. Random Forest
    logger.info("\n" + "-" * 40)
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    rf_results = evaluate_sklearn_model(rf_model, X_test, y_test, "Random Forest")
    results.append(rf_results)
    logger.info(f"RF: PR-AUC={rf_results['pr_auc']:.4f}, F1={rf_results['f1']:.4f}")
    
    # 3. XGBoost
    if HAS_XGBOOST:
        logger.info("\n" + "-" * 40)
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_model:
            xgb_results = evaluate_sklearn_model(xgb_model, X_test, y_test, "XGBoost")
            results.append(xgb_results)
            logger.info(f"XGB: PR-AUC={xgb_results['pr_auc']:.4f}, F1={xgb_results['f1']:.4f}")
    
    # 4. MLP
    logger.info("\n" + "-" * 40)
    mlp_scaler = StandardScaler()
    mlp_model = train_mlp(X_train, y_train, X_val, y_val, mlp_scaler)
    X_test_mlp = mlp_scaler.transform(X_test)
    mlp_results = evaluate_sklearn_model(mlp_model, X_test_mlp, y_test, "MLP")
    results.append(mlp_results)
    logger.info(f"MLP: PR-AUC={mlp_results['pr_auc']:.4f}, F1={mlp_results['f1']:.4f}")
    
    # 5. TemporalGAT (Our method)
    if not args.skip_gnn:
        logger.info("\n" + "-" * 40)
        # Reload sequences (they were modified)
        train_seq, val_seq, test_seq = create_temporal_splits(
            dataset, train_timesteps=42, val_timesteps=5, test_timesteps=2,
        )
        gat_results = train_temporal_gat(train_seq, val_seq, test_seq, device, args.epochs)
        results.append(gat_results)
        logger.info(f"GAT: PR-AUC={gat_results['pr_auc']:.4f}, F1={gat_results['f1']:.4f}")
    
    # Summary table
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Model':<25} {'PR-AUC':>10} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-" * 80)
    
    for r in results:
        logger.info(f"{r['model']:<25} {r['pr_auc']:>10.4f} {r['roc_auc']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
    
    # Save results
    with open(output_dir / 'baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = [r['model'] for r in results]
    pr_aucs = [r['pr_auc'] for r in results]
    f1s = [r['f1'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pr_aucs, width, label='PR-AUC', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Baseline Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.7)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
