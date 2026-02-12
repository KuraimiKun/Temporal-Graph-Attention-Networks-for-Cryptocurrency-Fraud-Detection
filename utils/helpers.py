"""
Utility Functions for the Fraud Detection Project
"""

import os
import random
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for each line
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{name}: {value:.4f}")
        else:
            lines.append(f"{prefix}{name}: {value}")
    return "\n".join(lines)


def create_experiment_dir(
    base_dir: str = "./experiments/",
    experiment_name: Optional[str] = None,
) -> Path:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional custom name (default: timestamp)
        
    Returns:
        Path to the created directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    exp_dir = Path(base_dir) / experiment_name
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    (exp_dir / "explanations").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    
    return exp_dir


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
):
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
    )


class AverageMeter:
    """
    Utility class to track running averages.
    
    Useful for tracking metrics during training.
    
    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch_loss in losses:
        ...     loss_meter.update(batch_loss)
        >>> print(f"Average loss: {loss_meter.avg:.4f}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_predictions(
    node_ids: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
):
    """
    Save model predictions to a CSV file.
    
    Args:
        node_ids: Array of node/transaction IDs
        probabilities: Fraud probability scores
        predictions: Binary predictions
        output_path: Path to save CSV
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'node_id': node_ids,
        'fraud_probability': probabilities,
        'prediction': predictions,
        'label': ['FRAUD' if p == 1 else 'LEGITIMATE' for p in predictions],
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with metric histories
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax1 = axes[0, 0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR-AUC
    ax2 = axes[0, 1]
    if 'val_pr_auc' in history:
        ax2.plot(history['val_pr_auc'], color='green')
        ax2.axhline(y=max(history['val_pr_auc']), color='r', linestyle='--',
                   label=f"Best: {max(history['val_pr_auc']):.4f}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Validation PR-AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score
    ax3 = axes[1, 0]
    if 'val_f1' in history:
        ax3.plot(history['val_f1'], color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Score')
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate
    ax4 = axes[1, 1]
    if 'lr' in history:
        ax4.plot(history['lr'], color='purple')
        ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    return fig
