# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weighted_ce(class_counts):
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()  # normalize
    return nn.CrossEntropyLoss(weight=weights)




class FocalLoss(nn.Module):
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
