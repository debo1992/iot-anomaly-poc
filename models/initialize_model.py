import torch
import torch.nn as nn
from models.lstm_basic import LSTMAnomalyClassifier
from models.cnn_basic import CNNAnomalyClassifier, AnomalyCNNDilation
from models.transformer_basic import TransformerAnomalyClassifier
from models.tcn_basic import TCNAnomalyClassifier


def my_model(config = None, input_dim=5):
    if config["model_type"] == "LSTM":
        model = LSTMAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "CNN":
        model = CNNAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "CNN_DILATION":
        model = AnomalyCNNDilation(input_dim=input_dim, num_classes=config.get("num_classes", 6))
    elif config["model_type"] == "TRANSFORMER":
        model = TransformerAnomalyClassifier(input_dim=input_dim)
    elif config["model_type"] == "TCN":
        model = TCNAnomalyClassifier(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    return model
    
