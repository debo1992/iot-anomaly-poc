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
