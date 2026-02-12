"""
Evaluation Metrics for Fraud Detection

This module implements evaluation metrics specifically designed for
imbalanced fraud detection tasks.

=== WHY PR-AUC OVER ACCURACY? ===

Consider a dataset with 1% fraud and 99% legitimate transactions:

Accuracy Problem:
- A "dumb" model that predicts ALL transactions as legitimate achieves 99% accuracy!
- But it catches ZERO frauds, making it completely useless.

ROC-AUC Problem:
- ROC-AUC uses True Positive Rate (TPR) and False Positive Rate (FPR)
- When negatives vastly outnumber positives, FPR can be misleadingly low
- A model with many false alarms may still have good ROC-AUC

PR-AUC Solution:
- Precision: Of all predicted frauds, how many are actually fraud?
- Recall: Of all actual frauds, how many did we catch?
- PR-AUC is sensitive to performance on the MINORITY class
- No credit for correctly classifying the easy majority class

In fraud detection:
- Precision matters: Don't waste investigator time on false alarms
- Recall matters: Don't miss actual frauds
- PR-AUC captures this tradeoff perfectly

=== RECOMMENDED METRICS ===

Primary: PR-AUC (Area Under Precision-Recall Curve)
Secondary: 
- F1-Score at optimal threshold
- Recall at fixed precision (e.g., R@P=0.9)
- Precision at fixed recall (e.g., P@R=0.9)

Avoid: Accuracy (misleading for imbalanced data)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class FraudMetrics:
    """
    Container for all fraud detection metrics.
    
    This class computes and stores all relevant metrics for fraud detection,
    with a focus on metrics that handle class imbalance appropriately.
    
    Attributes:
        pr_auc: Area Under Precision-Recall Curve (PRIMARY METRIC)
        roc_auc: Area Under ROC Curve
        f1: F1 score at optimal/specified threshold
        precision: Precision at threshold
        recall: Recall at threshold
        accuracy: Overall accuracy (use with caution!)
        threshold: Classification threshold used
        confusion_matrix: [[TN, FP], [FN, TP]]
        
    Example:
        >>> metrics = FraudMetrics.compute(y_true, y_scores)
        >>> print(f"PR-AUC: {metrics.pr_auc:.4f}")
        >>> print(f"Recall at P=0.9: {metrics.recall_at_precision(0.9):.4f}")
    """
    
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    threshold: float = 0.5
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    
    # Additional metrics
    avg_precision: float = 0.0  # Same as PR-AUC (sklearn convention)
    specificity: float = 0.0  # TN / (TN + FP)
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Precision-Recall curve data (for plotting)
    precisions: np.ndarray = field(default_factory=lambda: np.array([]))
    recalls: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @classmethod
    def compute(
        cls,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: Optional[float] = None,
        optimize_threshold: bool = True,
    ) -> 'FraudMetrics':
        """
        Compute all metrics from ground truth and prediction scores.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_scores: Predicted fraud probabilities (0 to 1)
            threshold: Fixed classification threshold
            optimize_threshold: Whether to find optimal threshold
            
        Returns:
            FraudMetrics instance with all computed metrics
        """
        metrics = cls()
        
        # Ensure numpy arrays
        if isinstance(y_true, Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_scores, Tensor):
            y_scores = y_scores.cpu().numpy()
        
        y_true = y_true.astype(int)
        
        # === Threshold-independent metrics ===
        
        # Precision-Recall curve and AUC
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_scores)
        metrics.pr_auc = auc(recalls, precisions)
        metrics.avg_precision = average_precision_score(y_true, y_scores)
        metrics.precisions = precisions
        metrics.recalls = recalls
        
        # ROC curve and AUC
        try:
            metrics.roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            # Can fail if only one class present
            metrics.roc_auc = 0.0
        
        # === Determine threshold ===
        if threshold is not None:
            metrics.threshold = threshold
        elif optimize_threshold:
            # Find threshold that maximizes F1
            metrics.threshold = find_optimal_threshold(
                y_true, y_scores, metric='f1'
            )
        else:
            metrics.threshold = 0.5
        
        # === Threshold-dependent metrics ===
        y_pred = (y_scores >= metrics.threshold).astype(int)
        
        metrics.f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics.precision = precision_score(y_true, y_pred, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, zero_division=0)
        metrics.accuracy = (y_true == y_pred).mean()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics.confusion_matrix = cm
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def recall_at_precision(self, target_precision: float) -> float:
        """
        Get recall at a specified precision level.
        
        Useful metric: "How many frauds can we catch while maintaining
        90% precision (i.e., 10% false alarm rate)?"
        """
        valid_idx = self.precisions >= target_precision
        if not valid_idx.any():
            return 0.0
        return self.recalls[valid_idx].max()
    
    def precision_at_recall(self, target_recall: float) -> float:
        """
        Get precision at a specified recall level.
        
        Useful metric: "What's our precision if we want to catch 90% of frauds?"
        """
        valid_idx = self.recalls >= target_recall
        if not valid_idx.any():
            return 0.0
        return self.precisions[valid_idx].max()
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            'pr_auc': self.pr_auc,
            'roc_auc': self.roc_auc,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'accuracy': self.accuracy,
            'threshold': self.threshold,
            'specificity': self.specificity,
            'avg_precision': self.avg_precision,
        }
    
    def __str__(self) -> str:
        """Pretty-print metrics."""
        lines = [
            "=" * 50,
            "FRAUD DETECTION METRICS",
            "=" * 50,
            f"PR-AUC (Primary):     {self.pr_auc:.4f}",
            f"ROC-AUC:              {self.roc_auc:.4f}",
            "-" * 50,
            f"Threshold:            {self.threshold:.4f}",
            f"F1 Score:             {self.f1:.4f}",
            f"Precision:            {self.precision:.4f}",
            f"Recall:               {self.recall:.4f}",
            f"Specificity:          {self.specificity:.4f}",
            "-" * 50,
            f"Accuracy:             {self.accuracy:.4f}  (NOTE: misleading for imbalanced data)",
            "-" * 50,
            "Confusion Matrix:",
            f"  TN={self.confusion_matrix[0,0]:5d}  FP={self.confusion_matrix[0,1]:5d}",
            f"  FN={self.confusion_matrix[1,0]:5d}  TP={self.confusion_matrix[1,1]:5d}",
            "=" * 50,
        ]
        return "\n".join(lines)


# =============================================================================
# Standalone Metric Functions
# =============================================================================

def compute_pr_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    This is the PRIMARY metric for fraud detection.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities for positive class
        
    Returns:
        PR-AUC score (0 to 1, higher is better)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    return auc(recalls, precisions)


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probabilities
        
    Returns:
        ROC-AUC score
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return 0.0


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics from binary predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions (not probabilities)
        
    Returns:
        Dictionary of metrics
    """
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': (y_true == y_pred).mean(),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f1',
) -> float:
    """
    Find the optimal classification threshold.
    
    Different optimization targets:
    - 'f1': Maximize F1 score (balance precision and recall)
    - 'youden': Maximize Youden's J (sensitivity + specificity - 1)
    - 'precision_recall_balance': Where precision equals recall
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        metric: Optimization target
        
    Returns:
        Optimal threshold value
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    if metric == 'f1':
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    
    elif metric == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Youden's J = TPR - FPR = Sensitivity + Specificity - 1
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    elif metric == 'precision_recall_balance':
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        # Find where precision ≈ recall
        diff = np.abs(precisions[:-1] - recalls[:-1])
        best_idx = np.argmin(diff)
        return thresholds[best_idx]
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, float]:
    """
    Plot the Precision-Recall curve.
    
    The PR curve shows the tradeoff between precision (y-axis) and 
    recall (x-axis) at different classification thresholds.
    
    For fraud detection:
    - Moving right (higher recall) = catch more frauds
    - Moving up (higher precision) = fewer false alarms
    - The curve's area (PR-AUC) summarizes overall performance
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        ax: Existing matplotlib axes
        
    Returns:
        (Figure, PR-AUC score)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recalls, precisions)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Plot PR curve
    ax.plot(recalls, precisions, 'b-', linewidth=2, 
            label=f'PR Curve (AUC = {pr_auc:.4f})')
    
    # Baseline: random classifier
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='r', linestyle='--', 
               label=f'Random (Fraud Rate = {baseline:.2%})')
    
    # Find and mark optimal F1 threshold
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    ax.scatter([best_recall], [best_precision], color='green', s=100, zorder=5,
               label=f'Best F1: P={best_precision:.2f}, R={best_recall:.2f}')
    
    ax.set_xlabel('Recall (Fraud Detection Rate)', fontsize=12)
    ax.set_ylabel('Precision (1 - False Alarm Rate)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")
    
    return fig, pr_auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, float]:
    """
    Plot the ROC (Receiver Operating Characteristic) curve.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        ax: Existing matplotlib axes
        
    Returns:
        (Figure, ROC-AUC score)
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Baseline: random classifier
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    
    # Mark Youden's J optimal point
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color='green', s=100, zorder=5,
               label=f"Youden's J: FPR={fpr[best_idx]:.2f}, TPR={tpr[best_idx]:.2f}")
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")
    
    return fig, roc_auc


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how metrics change with classification threshold.
    
    Useful for understanding the precision-recall tradeoff and
    selecting an appropriate threshold for deployment.
    """
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, Tensor):
        y_scores = y_scores.cpu().numpy()
    
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
    
    # Mark optimal F1
    best_idx = np.argmax(f1_scores)
    ax.axvline(x=thresholds[best_idx], color='green', linestyle='--', alpha=0.7)
    ax.scatter([thresholds[best_idx]], [f1_scores[best_idx]], 
               color='green', s=100, zorder=5)
    
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Metrics vs. Classification Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_metrics_summary_plot(
    metrics: FraudMetrics,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a comprehensive metrics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PR Curve
    ax1 = axes[0, 0]
    ax1.plot(metrics.recalls, metrics.precisions, 'b-', linewidth=2)
    ax1.fill_between(metrics.recalls, metrics.precisions, alpha=0.3)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall Curve (AUC = {metrics.pr_auc:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # Confusion Matrix Heatmap
    ax2 = axes[0, 1]
    cm = metrics.confusion_matrix
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted\nLegitimate', 'Predicted\nFraud'])
    ax2.set_yticklabels(['Actual\nLegitimate', 'Actual\nFraud'])
    
    # Add values to cells
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{cm[i, j]:d}', ha='center', va='center', fontsize=14)
    
    ax2.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax2)
    
    # Metrics Bar Chart
    ax3 = axes[1, 0]
    metric_names = ['PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall']
    metric_values = [metrics.pr_auc, metrics.roc_auc, metrics.f1, 
                     metrics.precision, metrics.recall]
    colors = ['gold', 'steelblue', 'green', 'coral', 'purple']
    
    bars = ax3.bar(metric_names, metric_values, color=colors, edgecolor='black')
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Score')
    ax3.set_title('Metric Comparison')
    
    for bar, val in zip(bars, metric_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10)
    
    # Text Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = str(metrics)
    ax4.text(0.1, 0.5, summary_text, family='monospace', fontsize=9,
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved metrics summary to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demonstrate metrics on synthetic data
    print("Fraud Detection Metrics Demo")
    print("=" * 50)
    
    # Simulate predictions
    np.random.seed(42)
    n_samples = 10000
    fraud_rate = 0.05  # 5% fraud
    
    # Ground truth
    y_true = np.random.choice([0, 1], size=n_samples, p=[1-fraud_rate, fraud_rate])
    
    # Simulated model predictions (fraud class gets higher scores)
    y_scores = np.random.beta(2, 5, size=n_samples)  # Base distribution
    y_scores[y_true == 1] = np.random.beta(5, 2, size=(y_true == 1).sum())  # Fraud higher
    
    # Compute metrics
    metrics = FraudMetrics.compute(y_true, y_scores, optimize_threshold=True)
    
    print(metrics)
    
    print(f"\nRecall at 90% Precision: {metrics.recall_at_precision(0.9):.4f}")
    print(f"Precision at 90% Recall: {metrics.precision_at_recall(0.9):.4f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = create_metrics_summary_plot(metrics, save_path=None)
    plt.show()
