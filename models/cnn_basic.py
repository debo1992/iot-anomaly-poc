import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAnomalyClassifier(nn.Module):
    def __init__(self, input_dim=5, num_classes=6, hidden_dim=64, dropout=0.2):
        super(CNNAnomalyClassifier, self).__init__()
        
        # 1D Convolutions across time dimension
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=4)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling across time
        x = x.mean(dim=-1)
        
        x = self.dropout(x)
        out = self.fc(x)
        return out



class AnomalyCNNDilation(nn.Module):
    def __init__(self, input_dim, num_classes, window_size=12):
        super(AnomalyCNNDilation, self).__init__()
        # input_dim = number of features (channels)
        # window_size = timesteps in sequence

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64,
                               kernel_size=5, padding=2)  # keeps same length
        self.bn1 = nn.BatchNorm1d(64)
        
        # Dilated conv to capture longer lag
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, dilation=2, padding=4)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global pooling compresses over time
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # input shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1).contiguous()  # -> (batch, features, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)   # (batch, channels, 1)
        x = x.squeeze(-1)         # (batch, channels)

        x = self.dropout(x)
        x = self.fc(x)
        return x


class DilatedCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, kernel_size=3):
        super(DilatedCNN, self).__init__()
        self.blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8]  # covers ~31 timesteps with k=3

        in_channels = input_dim
        for d in dilations:
            self.blocks.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size,
                          padding=d, dilation=d)
            )
            in_channels = hidden_dim

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        for conv in self.blocks:
            residual = x
            x = F.relu(conv(x))
            # match shapes for residual
            if residual.shape[1] != x.shape[1]:
                residual = nn.Conv1d(residual.shape[1], x.shape[1], 1)(residual)
            x = x + residual  # residual connection
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)
