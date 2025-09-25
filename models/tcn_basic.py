# models/tcn_anomaly.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Chop off extra padding at the end to keep sequence length constant."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.transpose(1, 2)  # -> (B, F, T) for Conv1d
        out = self.network(x)
        out = out.transpose(1, 2)  # back to (B, T, F)
        return out


class TCNAnomalyClassifier(nn.Module):
    def __init__(self, input_dim=5, num_classes=6, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TCN(input_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x shape: (B, T, F)
        out = self.tcn(x)              # (B, T, C)
        out = out[:, -1, :]            # take last timestep
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = TCNAnomalyClassifier()
    dummy = torch.randn(8, 100, 5)   # batch=8, seq_len=100, features=5
    out = model(dummy)
    print(out.shape)  # (8, 6)
