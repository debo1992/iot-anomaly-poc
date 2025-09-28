# datasets/anomaly_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from utils.data_preprocess import feature_engineer

class AnomalyDataset(Dataset):
    def __init__(self, df, window_size=12, feature_columns=None):
        """
        df: DataFrame with anomaly features + 'anomaly_class'
        window_size: number of timesteps per input sequence
        feature_columns: list of columns to use as input features
        """
        self.feature_columns = feature_columns or [col for col in df.columns if col != "anomaly_class"]
        self.X, self.y = self.create_sequences(df, window_size)

    def create_sequences(self, df, window_size):
        values = df[self.feature_columns].values
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



def make_balanced_loader(dataset, batch_size):
    labels = [y for _, y in dataset]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def load_dataset(config=None, train_df=None, val_df=None):
    train_df = feature_engineer(train_df)
    val_df = feature_engineer(val_df)
    feature_cols = train_df.columns[2:10].tolist()
    train_df = train_df[feature_cols]
    val_df = val_df[feature_cols]

    # Create dataset
    train_dataset = AnomalyDataset(train_df, window_size=config["window_size"])
    val_dataset = AnomalyDataset(val_df, window_size=config["window_size"])

    # ✅ Use balanced sampler for training
    if config.get("balanced_loader", False):
        train_loader = make_balanced_loader(train_dataset, batch_size=config["batch_size"])
    else:
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader

def get_calibration_loader(train_dataset, frac=0.1, batch_size=64, shuffle=True):
    """
    Create a calibration DataLoader from a fraction of the training dataset.
    """
    n_total = len(train_dataset)
    n_calib = max(1, int(n_total * frac))  # at least 1 sample
    indices = np.random.choice(n_total, n_calib, replace=False)
    calib_subset = Subset(train_dataset, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=shuffle)