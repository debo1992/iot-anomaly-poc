import torch
import torch.nn as nn
from models.lstm_basic import LSTMAnomalyClassifier
from models.cnn_basic import CNNAnomalyClassifier
from models.transformer_basic import TransformerAnomalyClassifier
from models.tcn_basic import TCNAnomalyClassifier


def my_model(config = None):
    if config["model_type"] == "LSTM":
        model = LSTMAnomalyClassifier()
    elif config["model_type"] == "CNN":
        model = CNNAnomalyClassifier()
    elif config["model_type"] == "TRANSFORMER":
        model = TransformerAnomalyClassifier()
    elif config["model_type"] == "TCN":
        model = TCNAnomalyClassifier()
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    return model
    
def my_loss(config = None):
    if config["loss_type"] == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif config["loss_type"] == "weighted_ce":
        weights = torch.tensor(config["class_weights"]).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif config["loss_type"] == "focal":
        from utils.losses import FocalLoss
        weights = None
        if config.get("class_weights") is not None:
            weights = torch.tensor(config["class_weights"]).float().to(device)
        criterion = FocalLoss(weight=weights, gamma=2)
    else:
        raise ValueError(f"Unknown loss type: {config['loss_type']}")
    return criterion