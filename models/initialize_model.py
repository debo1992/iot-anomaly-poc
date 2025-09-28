"""
Factory function to instantiate different anomaly detection models based on configuration.

Supported model types:
- LSTM: Recurrent LSTM-based classifier for sequential data.
- CNN: Basic 1D CNN classifier.
- CNN_DILATION: CNN with dilated convolutions.
- DilatedCNN: Residual CNN with multiple dilated convolution blocks.
- TRANSFORMER: Transformer-based sequence classifier.
- TCN: Temporal Convolutional Network classifier.

The function returns the appropriate PyTorch model instance with the given input dimension and optional number of classes.
"""

import torch
import torch.nn as nn
from models.lstm_basic import LSTMAnomalyClassifier
from models.cnn_basic import CNNAnomalyClassifier, AnomalyCNNDilation, DilatedCNN
from models.transformer_basic import TransformerAnomalyClassifier
from models.tcn_basic import TCNAnomalyClassifier


def my_model(config = None, input_dim=5):
    if config["model_type"] == "LSTM":
        model = LSTMAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "CNN":
        model = CNNAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "CNN_DILATION":
        model = AnomalyCNNDilation(input_dim=input_dim, num_classes=config.get("num_classes", 6))
    elif config["model_type"] == "DilatedCNN":
        model = DilatedCNN(input_dim=input_dim, num_classes=config.get("num_classes", 6))   
    elif config["model_type"] == "TRANSFORMER":
        model = TransformerAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "TCN":
        model = TCNAnomalyClassifier(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    return model
    
