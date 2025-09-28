"""
This module defines loss functions for training classification models, including:

- Weighted Cross-Entropy Loss: Computes class weights inversely proportional to class frequencies to handle class imbalance.
- Focal Loss: Focuses training on hard-to-classify examples by down-weighting easy examples, improving performance on imbalanced datasets.
- my_loss(config, device): Factory function that returns the appropriate loss function based on the configuration, supporting standard cross-entropy, weighted cross-entropy, and focal loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def get_weighted_ce(class_counts):
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()  # normalize
    return nn.CrossEntropyLoss(weight=weights)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

Down-weights easy examples and focuses training on hard misclassified ones,
commonly used in imbalanced classification problems.

Args:
    weight (torch.Tensor, optional): Class weights (1D tensor of shape [C]).
        Defaults to None.
    gamma (float, optional): Focusing parameter; larger values increase
        the penalty on hard misclassified samples. Defaults to 2.0.

Input:
    inputs (torch.Tensor): Logits of shape (N, C).
    targets (torch.Tensor): Ground truth class indices of shape (N,).

Returns:
    torch.Tensor: Scalar focal loss value.
"""
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
def my_loss(config = None, device='cpu'):
    if config["loss_type"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif config["loss_type"] == "weighted_ce":
        weights = torch.tensor(config["class_weights"]).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif config["loss_type"] == "focal":
        from models.losses import FocalLoss
        weights = None
        if config.get("class_weights") is not None:
            weights = torch.tensor(config["class_weights"]).float().to(device)
        criterion = FocalLoss(weight=weights, gamma=2)
    else:
        raise ValueError(f"Unknown loss type: {config['loss_type']}")
    return criterion
