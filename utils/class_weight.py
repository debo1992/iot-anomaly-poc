# utils/class_weights.py
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(train_labels):
    """
    Compute class weights for imbalanced classification.

    Args:
        train_labels (array-like): 1D array or list of labels from training data.
        device (str or torch.device): where to store the tensor ('cpu' or 'cuda').

    Returns:
        class_weights_tensor (torch.Tensor): tensor of class weights for PyTorch loss.
        classes (np.ndarray): array of class labels.
    """
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    
    return weights, classes
