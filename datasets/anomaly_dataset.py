# datasets/anomaly_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class AnomalyDataset(Dataset):
    def __init__(self, df, window_size=12):
        """
        df: DataFrame with columns:
            ['temperature_c','humidity_pct','fridge_power_w','front_door_open','fire_alarm','anomaly_class']
        window_size: number of timesteps per input sequence
        """
        self.X, self.y = self.create_sequences(df, window_size)

    def create_sequences(self, df, window_size):
        features = ["temperature_c", "humidity_pct", "fridge_power_w", "front_door_open", "fire_alarm"]
        values = df[features].values
        labels = df["anomaly_class"].values

        X, y = [], []
        for i in range(len(df) - window_size):
            seq_x = values[i:i+window_size]
            seq_y = labels[i+window_size-1]  # Use last step's label
            X.append(seq_x)
            y.append(seq_y)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
