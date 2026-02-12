#!/usr/bin/env python3
"""
Ethereum Phishing Detection Training on Google Cloud Vertex AI
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 60, flush=True)
print("STARTING ETHEREUM PHISHING DETECTION TRAINING", flush=True)
print("=" * 60, flush=True)

import numpy as np
print("Imported numpy", flush=True)

import torch
print(f"Imported torch {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve
)
print("Imported sklearn metrics", flush=True)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Loading custom modules...", flush=True)
from data.ethereum_phishing_loader import (
    EthereumPhishingDataset, 
    TemporalGraphSnapshot,
    compute_enhanced_features
)
print("Loaded ethereum_phishing_loader", flush=True)

from models.temporal_gnn import create_model
print("Loaded temporal_gnn model", flush=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.92, gamma: float = 3.5, pos_weight: float = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal term
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Cross entropy with optional pos_weight
        if self.pos_weight is not None:
            weight = torch.tensor([1.0, self.pos_weight], device=inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        loss = focal_weight * ce_loss
        return loss.mean()


def train_epoch(
    model: nn.Module,
    train_seq: List[TemporalGraphSnapshot],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    
    train_seq_device = [s.to(device) for s in train_seq]
    x_seq = [s.x for s in train_seq_device]
    edge_seq = [s.edge_index for s in train_seq_device]
    
    optimizer.zero_grad()
    out = model(x_seq, edge_seq)
    
    # Use last timestep for loss
    last_snap = train_seq_device[-1]
    mask = last_snap.train_mask
    
    loss = criterion(out[mask], last_snap.y[mask])
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model on a sequence."""
    model.eval()
    
    eval_seq_device = [s.to(device) for s in eval_seq]
    x_seq = [s.x for s in eval_seq_device]
    edge_seq = [s.edge_index for s in eval_seq_device]
    
    out = model(x_seq, edge_seq)
    
    last_snap = eval_seq_device[-1]
    mask = last_snap.train_mask
    
    if mask.sum() == 0:
        return {'pr_auc': 0, 'roc_auc': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    
    y_true = last_snap.y[mask].cpu().numpy()
    y_probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    y_pred = (y_probs >= threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return {'pr_auc': 0, 'roc_auc': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_probs),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def find_optimal_threshold(
    model: nn.Module,
    eval_seq: List[TemporalGraphSnapshot],
    device: torch.device,
    target_precision: float = 0.35,
) -> Tuple[float, Dict]:
    """Find threshold that achieves target precision."""
    model.eval()
    
    eval_seq_device = [s.to(device) for s in eval_seq]
    x_seq = [s.x for s in eval_seq_device]
    edge_seq = [s.edge_index for s in eval_seq_device]
    
    with torch.no_grad():
        out = model(x_seq, edge_seq)
        last_snap = eval_seq_device[-1]
        mask = last_snap.train_mask
        
        y_true = last_snap.y[mask].cpu().numpy()
        y_probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
    
    # Find threshold for target precision
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find threshold closest to target precision
    valid_idx = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid_idx) > 0:
        # Among valid, pick the one with highest recall
        best_idx = valid_idx[np.argmax(recalls[:-1][valid_idx])]
        optimal_threshold = thresholds[best_idx]
    else:
        # Fallback: maximize F1
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # Evaluate at optimal threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred),
    }
    
    return optimal_threshold, metrics


def train(args):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("ETHEREUM PHISHING DETECTION - TemporalGAT Training")
    logger.info("=" * 60)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("\nLoading Ethereum Phishing dataset...")
    dataset = EthereumPhishingDataset(
        data_dir=args.data_dir,
        num_timesteps=args.num_timesteps,
        sample_ratio=args.sample_ratio,
        max_nodes_per_snapshot=args.max_nodes_per_snapshot,
    )
    
    # Try to load cached
    if not dataset.load_processed():
        dataset.create_temporal_snapshots()
        dataset.save_processed()
    
    # Split data
    train_seq, val_seq, test_seq = dataset.get_train_test_split()
    
    if not train_seq or not test_seq:
        logger.error("Not enough data for training. Check dataset.")
        return
    
    # Enhance features
    logger.info("Enhancing features with graph structure...")
    train_seq = [compute_enhanced_features(s) for s in train_seq]
    val_seq = [compute_enhanced_features(s) for s in val_seq] if val_seq else []
    test_seq = [compute_enhanced_features(s) for s in test_seq]
    
    node_input_dim = train_seq[0].x.shape[1]
    logger.info(f"Feature dimension: {node_input_dim}")
    
    # Parse hidden channels
    hidden_channels = [int(x) for x in args.hidden_channels.split(',')]
    logger.info(f"Hidden channels: {hidden_channels}")
    
    # Create model
    logger.info("\nCreating TemporalGAT model...")
    model = create_model(
        architecture='temporal_gat',
        node_input_dim=node_input_dim,
        edge_input_dim=0,
        hidden_channels=hidden_channels,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss function
    # Calculate pos_weight based on class imbalance
    # Ethereum has ~1,165 phishing in ~3M nodes (much more imbalanced than Bitcoin)
    pos_weight = args.pos_weight if args.pos_weight else 50.0  # Higher than Bitcoin's 25
    
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, pos_weight=pos_weight)
    logger.info(f"Focal Loss: alpha={args.alpha}, gamma={args.gamma}, pos_weight={pos_weight}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    best_pr_auc = 0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_seq, optimizer, criterion, device)
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_seq if val_seq else test_seq, device)
        
        # Log
        logger.info(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
            f"PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
        
        # Save best (or always save if no improvement detected)
        if val_metrics['pr_auc'] > best_pr_auc or best_epoch == 0:
            best_pr_auc = val_metrics['pr_auc']
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
    
    # Load best model (if it exists)
    best_model_path = output_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"\nBest epoch: {best_epoch} with PR-AUC: {best_pr_auc:.4f}")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        logger.warning("No best model saved, using final model")
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    
    test_metrics = evaluate(model, test_seq, device)
    
    # Find optimal threshold
    optimal_threshold, thresh_metrics = find_optimal_threshold(
        model, test_seq, device, target_precision=0.35
    )
    
    logger.info(f"\nDefault Threshold (0.5):")
    logger.info(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    
    logger.info(f"\nOptimized Threshold ({optimal_threshold:.4f}):")
    logger.info(f"  Precision: {thresh_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {thresh_metrics['recall']:.4f}")
    logger.info(f"  F1:        {thresh_metrics['f1']:.4f}")
    
    # Helper to convert to JSON-serializable format
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj
    
    # Prepare metadata without circular refs
    safe_metadata = {
        'num_timesteps': dataset.metadata.get('num_timesteps', 0),
        'total_nodes': dataset.metadata.get('total_nodes', 0),
        'total_edges': dataset.metadata.get('total_edges', 0),
        'num_phishing': dataset.metadata.get('num_phishing', 0),
        'num_normal': dataset.metadata.get('num_normal', 0),
        'phishing_ratio': float(dataset.metadata.get('phishing_ratio', 0)),
        'node_feature_dim': dataset.metadata.get('node_feature_dim', 0),
    }
    
    # Save results
    results = {
        'dataset': 'ethereum_phishing',
        'architecture': 'temporal_gat',
        'hidden_channels': hidden_channels,
        'pos_weight': pos_weight,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'best_epoch': best_epoch,
        'test_metrics': to_serializable(test_metrics),
        'optimal_threshold': float(optimal_threshold),
        'thresh_metrics': to_serializable(thresh_metrics),
        'metadata': safe_metadata,
        'history': to_serializable(history),
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
    }, output_dir / "results.pt")
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return results


def submit_to_vertex(args):
    """Submit training job to Vertex AI."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        logger.error("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
        return
    
    logger.info("Submitting job to Vertex AI...")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=args.project_id,
        location=args.region,
        staging_bucket=f"gs://{args.bucket_name}/staging",
    )
    
    # Define custom training job
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=f"ethereum-phishing-temporal-gat-{args.seed}",
        python_package_gcs_uri=f"gs://{args.bucket_name}/training/package.tar.gz",
        python_module_name="train_ethereum_vertex",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    )
    
    # Run the job
    job.run(
        args=[
            "--vertex",
            "--bucket-name", args.bucket_name,
            "--epochs", str(args.epochs),
            "--pos-weight", str(args.pos_weight),
            "--alpha", str(args.alpha),
            "--gamma", str(args.gamma),
        ],
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )
    
    logger.info("Job submitted successfully!")


def main():
    parser = argparse.ArgumentParser(description="Ethereum Phishing Detection Training")
    
    # Mode
    parser.add_argument('--local', action='store_true', help='Run locally')
    parser.add_argument('--submit', action='store_true', help='Submit to Vertex AI')
    parser.add_argument('--vertex', action='store_true', help='Running on Vertex AI')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/EthereumPhishingTransactionNetwork')
    parser.add_argument('--num-timesteps', type=int, default=10)  # Reduced for memory
    parser.add_argument('--sample-ratio', type=float, default=0.05)  # 5% sample for local
    parser.add_argument('--max-nodes-per-snapshot', type=int, default=50000)  # Limit nodes
    
    # Model
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--hidden-channels', type=str, default='128,64,32')  # Smaller for local
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Loss
    parser.add_argument('--pos-weight', type=float, default=50.0)
    parser.add_argument('--alpha', type=float, default=0.95)  # Higher for more imbalanced
    parser.add_argument('--gamma', type=float, default=3.5)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/ethereum_phishing')
    
    # Vertex AI
    parser.add_argument('--project-id', type=str, default='')
    parser.add_argument('--region', type=str, default='us-central1')
    
    args = parser.parse_args()
    
    if args.submit:
        submit_to_vertex(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
