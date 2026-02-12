"""
Loss Functions for Imbalanced Fraud Detection

This module implements specialized loss functions for handling the severe
class imbalance in fraud detection (~1% fraud vs ~99% legitimate).

Problem with Standard Cross-Entropy:
- Treats all misclassifications equally
- Model can achieve high accuracy by predicting all samples as legitimate
- Fails to learn the minority (fraud) class

Solutions Implemented:
1. Focal Loss: Down-weights easy examples, focuses on hard ones
2. Weighted Cross-Entropy: Gives higher weight to minority class
3. Class-Balanced Loss: Weights inversely proportional to class frequency

Key Insight for Fraud Detection:
We care MORE about catching frauds (recall) than about false alarms (precision).
These loss functions help the model focus on correctly classifying fraud cases.
"""

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in fraud detection.
    
    Focal Loss was introduced for dense object detection but works
    excellently for fraud detection. It reduces the loss contribution
    from easy-to-classify examples, making the model focus on hard cases.
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
        - p_t: probability of correct class
        - α: class weight (higher for fraud class)
        - γ (gamma): focusing parameter (typically 2.0)
    
    When γ = 0: Focal Loss = Cross-Entropy Loss
    When γ > 0: Easy examples (high p_t) contribute less to loss
    
    Args:
        alpha: Weight for positive (fraud) class. Can be:
            - float: weight for positive class (1-alpha for negative)
            - Tensor: per-class weights [weight_neg, weight_pos]
        gamma: Focusing parameter. Higher = more focus on hard examples.
               Typical values: 1.0, 2.0, 5.0
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Smoothing factor for labels (0.0 to 0.1)
        
    Example:
        >>> loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
        >>> logits = model(x, edge_index)
        >>> loss = loss_fn(logits, labels)
        
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    
    def __init__(
        self,
        alpha: Union[float, Tensor] = 0.75,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        # Handle alpha (class weights)
        if isinstance(alpha, float):
            # Convert to [1-alpha, alpha] for [negative, positive]
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Predicted logits [N, num_classes]
            targets: Ground truth labels [N]
            mask: Optional boolean mask for valid samples [N]
                  (useful for ignoring unknown labels)
                  
        Returns:
            Scalar loss value (or per-sample if reduction='none')
        """
        num_classes = inputs.size(-1)
        
        # Move alpha to correct device
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # Apply mask if provided
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot
        
        # Get probability of true class
        p_t = (probs * targets_smooth).sum(dim=-1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Get alpha for each sample based on its class
        alpha_t = self.alpha[targets]
        
        # Compute focal cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for imbalanced classification.
    
    Simple but effective approach: assign higher weight to the
    minority (fraud) class to penalize misclassification more heavily.
    
    Weight Calculation Options:
    1. Inverse frequency: weight = 1 / class_count
    2. Inverse square root: weight = 1 / sqrt(class_count)
    3. Effective number: weight = (1 - β) / (1 - β^n) where β ≈ 0.9999
    
    Args:
        weight: Per-class weights [weight_neg, weight_pos]
                If None, computed from class counts
        class_counts: [count_neg, count_pos] for auto-computing weights
        weighting_method: 'inverse', 'inverse_sqrt', or 'effective'
        reduction: 'none', 'mean', or 'sum'
        
    Example:
        >>> # With known class distribution
        >>> loss_fn = WeightedCrossEntropyLoss(
        ...     class_counts=[95000, 5000],  # 5% fraud
        ...     weighting_method='inverse'
        ... )
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        class_counts: Optional[List[int]] = None,
        weighting_method: str = 'inverse',
        reduction: str = 'mean',
    ):
        super().__init__()
        
        if weight is not None:
            self.weight = weight
        elif class_counts is not None:
            self.weight = self._compute_weights(class_counts, weighting_method)
        else:
            self.weight = None
        
        self.reduction = reduction
    
    def _compute_weights(
        self,
        class_counts: List[int],
        method: str
    ) -> Tensor:
        """Compute class weights from counts."""
        counts = torch.tensor(class_counts, dtype=torch.float32)
        total = counts.sum()
        
        if method == 'inverse':
            # Inverse frequency weighting
            weights = total / (len(counts) * counts)
        
        elif method == 'inverse_sqrt':
            # Inverse square root - less aggressive
            weights = torch.sqrt(total / counts)
        
        elif method == 'effective':
            # Effective number of samples (Cui et al., 2019)
            beta = 0.9999
            effective_num = 1 - torch.pow(beta, counts)
            weights = (1 - beta) / effective_num
        
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize so weights sum to num_classes
        weights = weights / weights.sum() * len(counts)
        
        return weights
    
    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute weighted cross-entropy loss."""
        # Move weights to correct device
        weight = None
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
        
        # Apply mask if provided
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return F.cross_entropy(
            inputs, targets,
            weight=weight,
            reduction=self.reduction
        )


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples.
    
    This loss function addresses the problem that as more samples
    are added to a class, the additional benefit of each sample
    diminishes (diminishing returns).
    
    The effective number of samples for a class with n samples is:
        E_n = (1 - β^n) / (1 - β)
    
    where β ∈ [0, 1) controls the rate of diminishing returns.
    - β → 0: All samples equally important (standard re-weighting)
    - β → 1: Only unique samples contribute (like inverse frequency)
    
    This can be combined with Focal Loss for even better performance.
    
    Args:
        class_counts: [count_class_0, count_class_1, ...]
        beta: Hyperparameter for effective number (typically 0.9, 0.99, 0.999)
        loss_type: 'focal', 'cross_entropy', or 'sigmoid'
        gamma: Focal loss gamma (if loss_type='focal')
        
    Reference:
        Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", 
        CVPR 2019
    """
    
    def __init__(
        self,
        class_counts: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Compute effective number weights
        self.weights = self._compute_effective_weights(class_counts)
    
    def _compute_effective_weights(self, class_counts: List[int]) -> Tensor:
        """Compute weights based on effective number of samples."""
        counts = torch.tensor(class_counts, dtype=torch.float32)
        
        # Effective number of samples
        effective_num = 1 - torch.pow(self.beta, counts)
        
        # Weights are inverse of effective number
        weights = (1 - self.beta) / effective_num
        
        # Normalize
        weights = weights / weights.sum() * len(counts)
        
        return weights
    
    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute class-balanced loss."""
        # Move weights to device
        weights = self.weights.to(inputs.device)
        
        # Apply mask
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Get per-sample weights based on their class
        sample_weights = weights[targets]
        
        if self.loss_type == 'cross_entropy':
            loss = F.cross_entropy(inputs, targets, reduction='none')
        
        elif self.loss_type == 'focal':
            probs = F.softmax(inputs, dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            loss = focal_weight * ce_loss
        
        elif self.loss_type == 'sigmoid':
            # Binary classification with sigmoid
            loss = F.binary_cross_entropy_with_logits(
                inputs[:, 1], targets.float(), reduction='none'
            )
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply class-balanced weights
        weighted_loss = sample_weights * loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for fraud detection.
    
    In fraud detection, missing a fraud (False Negative) is typically
    more costly than a false alarm (False Positive). This loss
    allows asymmetric penalties:
    
    L = -γ * y * log(p) - (1-y) * log(1-p)
    
    where γ > 1 penalizes missing frauds more than false alarms.
    
    Args:
        fn_weight: Weight for False Negatives (missing frauds)
        fp_weight: Weight for False Positives (false alarms)
    """
    
    def __init__(
        self,
        fn_weight: float = 10.0,  # High cost for missing fraud
        fp_weight: float = 1.0,   # Lower cost for false alarm
        reduction: str = 'mean',
    ):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.reduction = reduction
    
    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute asymmetric loss."""
        if mask is not None:
            inputs = inputs[mask]
            targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Get probability of fraud (positive class)
        probs = F.softmax(inputs, dim=-1)[:, 1]
        
        # Separate positive (fraud) and negative (legitimate) samples
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        # Loss for positive samples (fraud)
        # FN = fraud but predicted as legitimate = low prob
        if pos_mask.sum() > 0:
            pos_loss = -self.fn_weight * torch.log(probs[pos_mask] + 1e-8)
        else:
            pos_loss = torch.tensor(0.0, device=inputs.device)
        
        # Loss for negative samples (legitimate)
        # FP = legitimate but predicted as fraud = high prob
        if neg_mask.sum() > 0:
            neg_loss = -self.fp_weight * torch.log(1 - probs[neg_mask] + 1e-8)
        else:
            neg_loss = torch.tensor(0.0, device=inputs.device)
        
        # Combine losses
        total_loss = torch.cat([pos_loss, neg_loss])
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


# =============================================================================
# Loss Function Factory
# =============================================================================

def create_loss_function(
    loss_type: str,
    class_counts: Optional[List[int]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions for fraud detection.
    
    Args:
        loss_type: One of 'focal', 'weighted_ce', 'class_balanced', 'asymmetric'
        class_counts: [num_legitimate, num_fraud] for weight computation
        **kwargs: Additional arguments for specific loss functions
        
    Returns:
        Configured loss function
        
    Example:
        >>> loss_fn = create_loss_function(
        ...     loss_type='focal',
        ...     class_counts=[95000, 5000],
        ...     gamma=2.0
        ... )
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.75)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type in ['weighted_ce', 'weighted_cross_entropy']:
        return WeightedCrossEntropyLoss(
            class_counts=class_counts,
            weighting_method=kwargs.get('weighting_method', 'inverse')
        )
    
    elif loss_type == 'class_balanced':
        if class_counts is None:
            raise ValueError("class_counts required for class_balanced loss")
        return ClassBalancedLoss(
            class_counts=class_counts,
            beta=kwargs.get('beta', 0.9999),
            loss_type=kwargs.get('base_loss', 'focal'),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            fn_weight=kwargs.get('fn_weight', 10.0),
            fp_weight=kwargs.get('fp_weight', 1.0)
        )
    
    elif loss_type in ['ce', 'cross_entropy']:
        return nn.CrossEntropyLoss()
    
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: 'focal', 'weighted_ce', 'class_balanced', 'asymmetric'"
        )


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions for Imbalanced Fraud Detection")
    print("=" * 60)
    
    # Simulate imbalanced data: 95% legitimate, 5% fraud
    num_samples = 1000
    num_fraud = 50
    
    logits = torch.randn(num_samples, 2)
    targets = torch.zeros(num_samples, dtype=torch.long)
    targets[:num_fraud] = 1  # First 50 are frauds
    
    class_counts = [num_samples - num_fraud, num_fraud]
    
    # Test each loss function
    losses = {
        'Focal Loss': FocalLoss(alpha=0.75, gamma=2.0),
        'Weighted CE': WeightedCrossEntropyLoss(class_counts=class_counts),
        'Class-Balanced': ClassBalancedLoss(class_counts=class_counts),
        'Asymmetric': AsymmetricLoss(fn_weight=10.0),
        'Standard CE': nn.CrossEntropyLoss(),
    }
    
    print(f"\nClass distribution: {class_counts[0]} legitimate, {class_counts[1]} fraud")
    print(f"Fraud ratio: {num_fraud/num_samples:.1%}\n")
    
    for name, loss_fn in losses.items():
        loss_value = loss_fn(logits, targets)
        print(f"{name:20s}: {loss_value.item():.4f}")
