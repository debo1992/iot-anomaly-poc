"""
LSTMAnomalyClassifier:
A multi-layer LSTM-based neural network for time-series anomaly classification.

- Takes input sequences of shape (batch_size, seq_len, input_dim).
- Processes sequences through stacked LSTM layers with dropout.
- Uses the last LSTM output (final time step) for classification via a fully connected layer.
- Outputs logits for multiple classes (default 6 classes).
"""
import torch.nn as nn

class LSTMAnomalyClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.2):
        super(LSTMAnomalyClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out
