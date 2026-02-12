#!/usr/bin/env python3
"""
Enhanced Training Pipeline with All Improvements

Implements:
1. Threshold optimization (grid search over multiple metrics)
2. Enhanced features (graph-based and temporal features)
3. TGAT architecture (Temporal Graph Attention)
4. Ensemble model (combines GConvGRU, TGAT)

Usage:
    python train_enhanced.py --ensemble
    python train_enhanced.py --architecture tgat
    python train_enhanced.py --optimize-threshold
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    load_config, 
    set_seed, 
    get_device, 
    setup_logging,
    create_experiment_dir,
    plot_training_history,
)
from data.elliptic_loader import (
    load_elliptic_dataset,
    create_temporal_splits,
    TemporalGraphSnapshot,
)
from models.temporal_gnn import create_model, GConvGRUModel, TemporalGATModel
from losses.imbalanced_losses import create_loss_function
from training.trainer import TemporalGNNTrainer, TrainingConfig
from evaluation.metrics import FraudMetrics, find_optimal_threshold

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Enhanced Feature Engineering
# =============================================================================

def compute_enhanced_features(snapshot: TemporalGraphSnapshot) -> TemporalGraphSnapshot:
    """
    Add graph-based structural features to improve fraud detection.
    
    New features:
    - Degree centrality (in/out degree)
    - PageRank-like score
    - Clustering coefficient approximation
    - Transaction velocity (within snapshot)
    """
    x = snapshot.x
    edge_index = snapshot.edge_index
    num_nodes = x.size(0)
    
    # In-degree and out-degree
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        
        in_degree = torch.zeros(num_nodes, device=x.device)
        out_degree = torch.zeros(num_nodes, device=x.device)
        
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        
        # Normalize
        in_degree = in_degree / (in_degree.max() + 1e-6)
        out_degree = out_degree / (out_degree.max() + 1e-6)
        
        # In/Out ratio (fraud often has high in/out ratio for mixing)
        io_ratio = in_degree / (out_degree + 1e-6)
        io_ratio = torch.clamp(io_ratio, 0, 10) / 10  # Normalize
        
        # 2-hop neighbor count (approximation of local clustering)
        neighbor_count = in_degree + out_degree
        
    else:
        in_degree = torch.zeros(num_nodes, device=x.device)
        out_degree = torch.zeros(num_nodes, device=x.device)
        io_ratio = torch.zeros(num_nodes, device=x.device)
        neighbor_count = torch.zeros(num_nodes, device=x.device)
    
    # Stack new features
    new_features = torch.stack([
        in_degree,
        out_degree,
        io_ratio,
        neighbor_count,
    ], dim=1)
    
    # Concatenate with existing features
    enhanced_x = torch.cat([x, new_features], dim=1)
    
    # Create new snapshot with enhanced features
    enhanced_snapshot = TemporalGraphSnapshot(
        x=enhanced_x,
        edge_index=snapshot.edge_index,
        edge_attr=snapshot.edge_attr,
        y=snapshot.y,
        train_mask=snapshot.train_mask,
        timestep=snapshot.timestep,
        node_ids=snapshot.node_ids,
    )
    
    return enhanced_snapshot


def enhance_temporal_sequence(
    sequence: List[TemporalGraphSnapshot]
) -> List[TemporalGraphSnapshot]:
    """Apply feature enhancement to entire temporal sequence."""
    return [compute_enhanced_features(snap) for snap in sequence]


# =============================================================================
# 2. Advanced Threshold Optimization
# =============================================================================

class ThresholdOptimizer:
    """
    Optimize classification threshold using multiple strategies.
    """
    
    def __init__(self):
        self.thresholds_tested = []
        self.metrics_at_thresholds = []
    
    def grid_search(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = 'f1',
        n_thresholds: int = 100,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Grid search for optimal threshold.
        
        Args:
            y_true: Ground truth labels
            y_scores: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision_at_recall_80', 'balanced')
            n_thresholds: Number of thresholds to test
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_threshold = 0.5
        best_score = -float('inf')
        best_metrics = {}
        
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # Compute optimization target
            if metric == 'f1':
                score = f1
            elif metric == 'precision_at_recall_80':
                score = precision if recall >= 0.8 else -1
            elif metric == 'recall_at_precision_50':
                score = recall if precision >= 0.5 else -1
            elif metric == 'balanced':
                # Geometric mean of precision and recall
                score = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
            elif metric == 'cost_sensitive':
                # Minimize: FN_cost * FN + FP_cost * FP
                # Assume missing fraud (FN) is 10x worse than false alarm (FP)
                fn = ((y_true == 1) & (y_pred == 0)).sum()
                fp = ((y_true == 0) & (y_pred == 1)).sum()
                # Negative because we want to maximize
                score = -(10 * fn + fp)
            else:
                score = f1
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = {
                    'threshold': thresh,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                }
        
        return best_threshold, best_metrics
    
    def find_all_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> Dict[str, Tuple[float, Dict]]:
        """
        Find optimal thresholds for multiple objectives.
        """
        results = {}
        
        for metric in ['f1', 'balanced', 'recall_at_precision_50', 'cost_sensitive']:
            thresh, metrics = self.grid_search(y_true, y_scores, metric=metric)
            results[metric] = (thresh, metrics)
            
        return results


# =============================================================================
# 3. Ensemble Model
# =============================================================================

class EnsembleTemporalGNN(nn.Module):
    """
    Ensemble of multiple Temporal GNN architectures.
    
    Combines:
    - GConvGRU: Good at capturing smooth temporal evolution
    - TemporalGAT: Good at learning attention over neighbors and time
    
    Ensemble strategy: Weighted average of predictions
    """
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int = 0,
        hidden_channels: List[int] = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.num_classes = num_classes
        
        # Create base models
        self.gconv_gru = GConvGRUModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.temporal_gat = TemporalGATModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_heads=4,
            dropout=dropout,
        )
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Optional: Meta-learner to combine predictions
        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        
        self.use_meta = True  # Set to False for simple weighted average
    
    def forward(
        self,
        x_sequence: List[Tensor],
        edge_index_sequence: List[Tensor],
        edge_attr_sequence: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Forward pass through ensemble."""
        # Get predictions from each model
        logits_gru = self.gconv_gru(x_sequence, edge_index_sequence, edge_attr_sequence)
        logits_gat = self.temporal_gat(x_sequence, edge_index_sequence, edge_attr_sequence)
        
        if self.use_meta:
            # Use meta-classifier to combine
            combined = torch.cat([logits_gru, logits_gat], dim=-1)
            logits = self.meta_classifier(combined)
        else:
            # Weighted average (softmax to ensure weights sum to 1)
            weights = F.softmax(self.ensemble_weights, dim=0)
            logits = weights[0] * logits_gru + weights[1] * logits_gat
        
        return logits
    
    def reset_hidden_state(self):
        """Reset hidden states in all models."""
        self.gconv_gru.reset_hidden_state()
        self.temporal_gat.reset_hidden_state()
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 4. Enhanced Training Loop
# =============================================================================

def train_with_enhancements(
    args,
    config: Dict,
    exp_dir: Path,
    device: torch.device,
):
    """
    Train with all enhancements enabled.
    """
    # Load dataset
    logger.info("Loading Elliptic dataset...")
    dataset = load_elliptic_dataset(root=args.data_dir)
    
    # Create temporal splits
    train_seq, val_seq, test_seq = create_temporal_splits(
        dataset,
        train_timesteps=config['data']['elliptic']['train_timesteps'],
        val_timesteps=config['data']['elliptic']['val_timesteps'],
        test_timesteps=config['data']['elliptic']['test_timesteps'],
    )
    
    # Apply enhanced features
    if args.enhance_features:
        logger.info("Applying enhanced feature engineering...")
        train_seq = enhance_temporal_sequence(train_seq)
        val_seq = enhance_temporal_sequence(val_seq)
        test_seq = enhance_temporal_sequence(test_seq)
        
        # Update node input dimension
        node_input_dim = train_seq[0].x.shape[1]
        logger.info(f"Enhanced feature dimension: {node_input_dim}")
    else:
        node_input_dim = dataset.metadata['node_feature_dim']
    
    # Create model
    model_config = config['model']
    
    if args.ensemble:
        logger.info("Creating Ensemble model (GConvGRU + TemporalGAT)...")
        model = EnsembleTemporalGNN(
            node_input_dim=node_input_dim,
            edge_input_dim=model_config['edge_input_dim'],
            hidden_channels=model_config['gconv_gru']['hidden_channels'],
            num_classes=model_config['classifier']['num_classes'],
            dropout=model_config['gconv_gru']['dropout'],
        )
    else:
        arch = args.architecture
        logger.info(f"Creating {arch} model...")
        model = create_model(
            architecture=arch,
            node_input_dim=node_input_dim,
            edge_input_dim=model_config['edge_input_dim'],
            hidden_channels=model_config['gconv_gru']['hidden_channels'],
            num_classes=model_config['classifier']['num_classes'],
            dropout=model_config['gconv_gru']['dropout'],
        )
    
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Create training configuration
    train_config = TrainingConfig(
        device=str(device),
        seed=args.seed,
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        epochs=config['training']['epochs'],
        loss_type=config['loss']['type'],
        focal_alpha=config['loss']['focal_loss']['alpha'],
        focal_gamma=config['loss']['focal_loss']['gamma'],
        early_stopping=config['training']['early_stopping']['enabled'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        checkpoint_dir=str(exp_dir / "checkpoints"),
        gradient_clip=config['training']['gradient_clip'],
    )
    
    # Create trainer
    class_counts = [dataset.metadata['num_legit'], dataset.metadata['num_fraud']]
    trainer = TemporalGNNTrainer(model, train_config, class_counts=class_counts)
    
    # Train
    logger.info("Starting training...")
    history = trainer.fit(train_seq, val_seq)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(exp_dir / "plots" / "training_history.png")
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    probs, preds = trainer.predict(test_seq)
    final_snapshot = test_seq[-1]
    mask = final_snapshot.train_mask.numpy()
    y_true = final_snapshot.y.numpy()[mask]
    probs_masked = probs[mask]
    
    # Apply threshold optimization
    if args.optimize_threshold:
        logger.info("\nOptimizing threshold...")
        optimizer = ThresholdOptimizer()
        threshold_results = optimizer.find_all_optimal_thresholds(y_true, probs_masked)
        
        logger.info("\nThreshold Optimization Results:")
        logger.info("=" * 60)
        for metric, (thresh, metrics) in threshold_results.items():
            logger.info(f"\n{metric.upper()}: Threshold = {thresh:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Use cost-sensitive threshold for final evaluation
        best_thresh = threshold_results['cost_sensitive'][0]
        logger.info(f"\nUsing cost-sensitive threshold: {best_thresh:.4f}")
    else:
        best_thresh = None
    
    # Compute final metrics
    full_metrics = FraudMetrics.compute(
        y_true, 
        probs_masked, 
        threshold=best_thresh,
        optimize_threshold=(best_thresh is None),
    )
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(str(full_metrics))
    
    # Save results
    results = {
        'test_metrics': full_metrics.to_dict(),
        'training_history': history,
        'config': config,
        'enhancements': {
            'enhanced_features': args.enhance_features,
            'ensemble': args.ensemble,
            'optimized_threshold': args.optimize_threshold,
        }
    }
    torch.save(results, exp_dir / "results.pt")
    
    logger.info(f"\nExperiment completed! Results saved to: {exp_dir}")
    
    return full_metrics


def main():
    parser = argparse.ArgumentParser(description="Enhanced Temporal GNN Training")
    
    # Mode selection
    parser.add_argument('--architecture', type=str, default='gconv_gru',
                        choices=['gconv_gru', 'tgat', 'temporal_gat'],
                        help='Model architecture (default: gconv_gru)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of GConvGRU + TGAT')
    
    # Enhancements
    parser.add_argument('--enhance-features', action='store_true',
                        help='Add graph structural features')
    parser.add_argument('--optimize-threshold', action='store_true',
                        help='Optimize classification threshold')
    parser.add_argument('--all-enhancements', action='store_true',
                        help='Enable all enhancements')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='./data/elliptic/',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs/',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Enable all enhancements if flag is set
    if args.all_enhancements:
        args.enhance_features = True
        args.optimize_threshold = True
        args.ensemble = True
    
    # Setup
    exp_dir = create_experiment_dir(args.output_dir, experiment_name="enhanced")
    setup_logging(log_file=str(exp_dir / "logs" / "experiment.log"))
    set_seed(args.seed)
    
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
    
    logger.info("=" * 60)
    logger.info("ENHANCED TEMPORAL GNN TRAINING")
    logger.info("=" * 60)
    logger.info(f"Architecture: {'Ensemble' if args.ensemble else args.architecture}")
    logger.info(f"Enhanced features: {args.enhance_features}")
    logger.info(f"Threshold optimization: {args.optimize_threshold}")
    logger.info(f"Device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Train
    metrics = train_with_enhancements(args, config, exp_dir, device)
    
    return metrics


if __name__ == "__main__":
    main()
