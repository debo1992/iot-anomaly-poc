"""
Transformer-based Anomaly Classifier

- Uses a learnable positional embedding and a Transformer encoder stack to model sequential data.
- Projects input features into a d_model-dimensional embedding space.
- Applies multi-head self-attention to capture temporal dependencies.
- Uses the last timestep's output embedding for classification via a fully connected head.
- Designed for multivariate time series anomaly classification with configurable depth and width.

Inputs:
- x: Tensor of shape (batch_size, sequence_length, input_dim)

Outputs:
- logits for num_classes (batch_size, num_classes)
"""

import torch
import torch.nn as nn

class TransformerAnomalyClassifier(nn.Module):
    def __init__(
        self,
        input_dim=5,         # features per timestep
        num_classes=6,       # output classes
        d_model=64,          # hidden size of embeddings
        nhead=4,             # number of attention heads
        num_layers=2,        # number of transformer encoder layers
        dim_feedforward=128, # FFN hidden dim
        dropout=0.1,
    ):
        super().__init__()

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable, simple version)
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model))  
        # 500 = max sequence length; adjust if you expect longer sequences

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, F)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        B, T, _ = x.size()

        # Project to embedding dimension
        x = self.input_proj(x)

        # Add position embeddings (trim to seq_len)
        pos = self.pos_embedding[:, :T, :]
        x = x + pos

        # Transformer encoder
        out = self.transformer_encoder(x)

        # Pooling: take the last timestep (like LSTM)
        # Or you can use mean pooling across time
        out = out[:, -1, :]

        # Classify
        return self.fc(out)


if __name__ == "__main__":
    model = TransformerAnomalyClassifier()
    dummy = torch.randn(8, 100, 5)  # batch=8, seq_len=100, features=5
    out = model(dummy)
    print(out.shape)  # (8, 6)
