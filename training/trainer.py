"""
Training Pipeline for Temporal GNN Fraud Detection

This module provides a complete training pipeline including:
- Training loop with temporal graph sequences
- Validation with PR-AUC monitoring
- Early stopping to prevent overfitting
- Learning rate scheduling
- Checkpointing and model selection
- Logging and experiment tracking
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.temporal_gnn import TemporalGNN
from losses.imbalanced_losses import create_loss_function
from evaluation.metrics import FraudMetrics, compute_pr_auc
from data.elliptic_loader import TemporalGraphSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for training the Temporal GNN.
    
    Organized into logical groups:
    - Model configuration
    - Optimization settings
    - Training loop parameters
    - Checkpointing and logging
    """
    
    # === Model ===
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # === Optimization ===
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    
    # === Loss ===
    loss_type: str = "focal"  # "focal", "weighted_ce", "class_balanced"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    
    # === Training Loop ===
    epochs: int = 100
    gradient_clip: float = 1.0
    
    # === Learning Rate Scheduler ===
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # === Early Stopping ===
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_pr_auc"
    early_stopping_mode: str = "max"  # "max" for PR-AUC, "min" for loss
    
    # === Checkpointing ===
    checkpoint_dir: str = "./checkpoints/"
    save_best_only: bool = True
    save_every_n_epochs: int = 5
    
    # === Logging ===
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = "ethereum-fraud-tgn"
    
    def __post_init__(self):
        """Create directories after initialization."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered! Best: {self.best_value:.4f} at epoch {self.best_epoch}"
                )
        
        return self.should_stop


class TemporalGNNTrainer:
    """
    Complete training pipeline for Temporal GNN fraud detection.
    
    Handles:
    - Processing sequences of temporal graph snapshots
    - Computing loss on labeled nodes only
    - Tracking PR-AUC for model selection
    - Early stopping and learning rate scheduling
    - Model checkpointing
    
    Example:
        >>> from models import create_model
        >>> from data import load_elliptic_dataset, create_temporal_splits
        >>> 
        >>> # Load data
        >>> dataset = load_elliptic_dataset()
        >>> train_seq, val_seq, test_seq = create_temporal_splits(dataset)
        >>> 
        >>> # Create model
        >>> model = create_model('gconv_gru', node_input_dim=166)
        >>> 
        >>> # Train
        >>> config = TrainingConfig(epochs=50, learning_rate=0.001)
        >>> trainer = TemporalGNNTrainer(model, config)
        >>> history = trainer.fit(train_seq, val_seq)
        >>> 
        >>> # Evaluate
        >>> test_metrics = trainer.evaluate(test_seq)
        >>> print(f"Test PR-AUC: {test_metrics.pr_auc:.4f}")
    """
    
    def __init__(
        self,
        model: TemporalGNN,
        config: TrainingConfig,
        class_counts: Optional[List[int]] = None,
    ):
        self.model = model
        self.config = config
        self.class_counts = class_counts
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize components
        self._init_loss()
        self._init_optimizer()
        self._init_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_pr_auc': [],
            'val_f1': [],
            'lr': [],
        }
        
        # Early stopping
        if config.early_stopping:
            self.early_stopper = EarlyStopping(
                patience=config.early_stopping_patience,
                mode=config.early_stopping_mode,
            )
        else:
            self.early_stopper = None
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {model.get_num_params():,}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _init_loss(self):
        """Initialize loss function."""
        self.loss_fn = create_loss_function(
            loss_type=self.config.loss_type,
            class_counts=self.class_counts,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize PR-AUC
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                min_lr=self.config.min_lr,
                verbose=True,
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        else:
            self.scheduler = None
    
    def fit(
        self,
        train_sequence: List[TemporalGraphSnapshot],
        val_sequence: List[TemporalGraphSnapshot],
    ) -> Dict[str, List[float]]:
        """
        Train the model on temporal graph sequences.
        
        Args:
            train_sequence: List of temporal snapshots for training
            val_sequence: List of temporal snapshots for validation
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Train sequence: {len(train_sequence)} timesteps")
        logger.info(f"Val sequence: {len(val_sequence)} timesteps")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch(train_sequence)
            
            # Validation
            val_metrics = self.evaluate(val_sequence)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_pr_auc'].append(val_metrics['pr_auc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            self._log_epoch(epoch, train_loss, val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['pr_auc'])
                else:
                    self.scheduler.step()
            
            # Checkpointing
            if val_metrics['pr_auc'] > self.best_metric:
                self.best_metric = val_metrics['pr_auc']
                self._save_checkpoint('best_model.pt', val_metrics)
            
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f'model_epoch_{epoch}.pt', val_metrics)
            
            # Early stopping
            if self.early_stopper is not None:
                if self.early_stopper(val_metrics['pr_auc'], epoch):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best Val PR-AUC: {self.best_metric:.4f}")
        
        return self.history
    
    def _train_epoch(
        self,
        train_sequence: List[TemporalGraphSnapshot],
    ) -> float:
        """
        Train for one epoch on temporal sequence.
        
        For temporal GNNs, we process the entire sequence as one "batch"
        because the model maintains hidden state across timesteps.
        Uses multi-step training with temporal consistency regularization.
        """
        self.model.train()
        self.model.reset_hidden_state()
        
        total_loss = 0.0
        total_samples = 0
        
        # Prepare sequence data
        x_sequence = []
        edge_index_sequence = []
        edge_attr_sequence = []
        y_sequence = []
        mask_sequence = []
        
        for snapshot in train_sequence:
            x_sequence.append(snapshot.x.to(self.device))
            edge_index_sequence.append(snapshot.edge_index.to(self.device))
            edge_attr_sequence.append(
                snapshot.edge_attr.to(self.device) if snapshot.edge_attr is not None else None
            )
            y_sequence.append(snapshot.y.to(self.device))
            mask_sequence.append(snapshot.train_mask.to(self.device))
        
        # Multi-step training: compute loss on multiple timesteps
        # This helps the model learn more robust temporal patterns
        self.optimizer.zero_grad()
        
        accumulated_loss = 0.0
        num_steps = min(5, len(train_sequence))  # Use last 5 timesteps
        step_weight = 1.0 / num_steps
        
        for step in range(num_steps):
            t = len(train_sequence) - num_steps + step
            
            # Forward pass up to timestep t
            subsequence_x = x_sequence[:t+1]
            subsequence_edge = edge_index_sequence[:t+1]
            subsequence_attr = edge_attr_sequence[:t+1]
            
            self.model.reset_hidden_state()
            logits = self.model(subsequence_x, subsequence_edge, subsequence_attr)
            
            # Compute loss on labeled nodes at this timestep
            step_y = y_sequence[t]
            step_mask = mask_sequence[t]
            
            step_loss = self.loss_fn(logits, step_y, mask=step_mask)
            accumulated_loss = accumulated_loss + step_weight * step_loss
        
        # Backward pass
        accumulated_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        self.global_step += 1
        
        total_loss = accumulated_loss.item()
        
        return total_loss
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_sequence: List[TemporalGraphSnapshot],
    ) -> Dict[str, float]:
        """
        Evaluate model on a temporal sequence.
        
        Args:
            eval_sequence: List of temporal snapshots
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.model.reset_hidden_state()
        
        # Prepare data
        x_sequence = [s.x.to(self.device) for s in eval_sequence]
        edge_index_sequence = [s.edge_index.to(self.device) for s in eval_sequence]
        edge_attr_sequence = [
            s.edge_attr.to(self.device) if s.edge_attr is not None else None
            for s in eval_sequence
        ]
        
        # Forward pass
        logits = self.model(x_sequence, edge_index_sequence, edge_attr_sequence)
        
        # Get labels and mask from final timestep
        final_snapshot = eval_sequence[-1]
        y_true = final_snapshot.y.to(self.device)
        mask = final_snapshot.train_mask.to(self.device)
        
        # Filter to labeled nodes
        logits_masked = logits[mask]
        y_true_masked = y_true[mask]
        
        # Compute loss
        loss = self.loss_fn(logits_masked, y_true_masked).item()
        
        # Get probabilities for fraud class
        probs = torch.softmax(logits_masked, dim=-1)[:, 1]
        
        # Compute metrics
        metrics = FraudMetrics.compute(
            y_true=y_true_masked.cpu().numpy(),
            y_scores=probs.cpu().numpy(),
            optimize_threshold=True,
        )
        
        return {
            'loss': loss,
            'pr_auc': metrics.pr_auc,
            'roc_auc': metrics.roc_auc,
            'f1': metrics.f1,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'threshold': metrics.threshold,
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
    ):
        """Log training progress."""
        lr = self.optimizer.param_groups[0]['lr']
        
        log_str = (
            f"Epoch {epoch:3d}/{self.config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.2e}"
        )
        logger.info(log_str)
        
        # W&B logging (if enabled)
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_pr_auc': val_metrics['pr_auc'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'learning_rate': lr,
                })
            except ImportError:
                pass
    
    def _save_checkpoint(
        self,
        filename: str,
        metrics: Dict[str, float],
    ):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
        
        return checkpoint.get('metrics', {})
    
    @torch.no_grad()
    def predict(
        self,
        sequence: List[TemporalGraphSnapshot],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a temporal sequence.
        
        Args:
            sequence: List of temporal snapshots
            
        Returns:
            Tuple of (fraud_probabilities, predicted_labels)
        """
        self.model.eval()
        self.model.reset_hidden_state()
        
        x_sequence = [s.x.to(self.device) for s in sequence]
        edge_index_sequence = [s.edge_index.to(self.device) for s in sequence]
        edge_attr_sequence = [
            s.edge_attr.to(self.device) if s.edge_attr is not None else None
            for s in sequence
        ]
        
        logits = self.model(x_sequence, edge_index_sequence, edge_attr_sequence)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        
        # Use best threshold from training if available
        threshold = self.history.get('best_threshold', 0.5)
        predictions = (probs >= threshold).astype(int)
        
        return probs, predictions


def train_model(
    model: TemporalGNN,
    train_sequence: List[TemporalGraphSnapshot],
    val_sequence: List[TemporalGraphSnapshot],
    config: Optional[TrainingConfig] = None,
    **kwargs
) -> Tuple[TemporalGNNTrainer, Dict[str, List[float]]]:
    """
    Convenience function to train a Temporal GNN model.
    
    Args:
        model: TemporalGNN model instance
        train_sequence: Training temporal snapshots
        val_sequence: Validation temporal snapshots
        config: Training configuration
        **kwargs: Override config parameters
        
    Returns:
        Tuple of (trainer, training_history)
    """
    if config is None:
        config = TrainingConfig(**kwargs)
    else:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    trainer = TemporalGNNTrainer(model, config)
    history = trainer.fit(train_sequence, val_sequence)
    
    return trainer, history


if __name__ == "__main__":
    # Quick test of training pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("Training Pipeline Test")
    print("=" * 50)
    
    # This would be used with actual data:
    # from data import load_elliptic_dataset, create_temporal_splits
    # from models import create_model
    #
    # dataset = load_elliptic_dataset()
    # train_seq, val_seq, test_seq = create_temporal_splits(dataset)
    # model = create_model('gconv_gru', node_input_dim=166)
    # trainer, history = train_model(model, train_seq, val_seq, epochs=50)
    # test_metrics = trainer.evaluate(test_seq)
    
    print("\nTo train a model, use:")
    print("  trainer, history = train_model(model, train_seq, val_seq)")
