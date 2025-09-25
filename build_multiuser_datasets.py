# scripts/build_multiuser_datasets.py
import os
import numpy as np
import pandas as pd
from datasets.generate_data import generate_user_data

def build_multiuser_datasets(
    train_users=80,
    val_users=20,
    start_date="2025-01-01",
    days=180,
    freq="1H",
    seed=123,
    output_dir="datasets/data"
):
    """
    Generate N users with per-user randomized base & noise/drift parameters.
    Saves per-user CSVs under output_dir/train and output_dir/val, and also
    concatenated train_all.csv and val_all.csv.
    """
    np.random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dfs = []
    val_dfs = []

    total_users = train_users + val_users
    for uid in range(1, total_users + 1):
        # Randomize per-user base & noise/drift parameters
        base_temp = 21.0 + np.random.uniform(-3.0, 3.0)         # user's baseline ±3°C
        base_temp_amp = np.random.uniform(1.5, 3.0)            # circadian amplitude
        base_humidity = 45.0 + np.random.uniform(-8.0, 10.0)   # user/habitat differences
        fridge_base = 150.0 + np.random.uniform(-25.0, 25.0)

        temp_drift = np.random.uniform(0.02, 0.08)             # °C/week
        humid_drift = np.random.uniform(0.03, 0.2)             # %RH/week

        noise_temp = np.random.uniform(0.15, 0.6)              # std dev °C
        noise_humid = np.random.uniform(1.0, 4.0)              # std dev %RH
        noise_fridge = np.random.uniform(0.5, 4.0)             # std dev W

        dropout_prob = np.random.choice([0.0, 0.0005, 0.001, 0.002])  # some users more flaky
        dropout_mean_duration = int(np.random.choice([1, 2, 3, 6]))

        # deterministic seed per-user for reproducibility
        user_seed = np.random.randint(0, 2**31 - 1)

        df, anomalies = generate_user_data(
            user_id=f"user_{uid:03d}",
            start_date=start_date,
            days=days,
            freq=freq,
            base_temp=base_temp,
            base_temp_amp=base_temp_amp,
            base_humidity=base_humidity,
            fridge_base=fridge_base,
            temp_drift_per_week=temp_drift,
            humid_drift_per_week=humid_drift,
            noise_temp=noise_temp,
            noise_humid=noise_humid,
            noise_fridge=noise_fridge,
            dropout_prob=dropout_prob,
            dropout_mean_duration=dropout_mean_duration,
            seed=user_seed
        )

        # Save per-user CSV with metadata columns included
        if uid <= train_users:
            path = os.path.join(train_dir, f"user_{uid:03d}.csv")
            train_dfs.append(df)
        else:
            path = os.path.join(val_dir, f"user_{uid:03d}.csv")
            val_dfs.append(df)
        df.to_csv(path, index=False)

    # Concatenate and save
    train_all = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    val_all = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    train_all.to_csv(os.path.join(output_dir, "train_all.csv"), index=False)
    val_all.to_csv(os.path.join(output_dir, "val_all.csv"), index=False)

    print("Finished generating datasets")
    print(f"- Train users: {train_users}, Val users: {val_users}")
    if not train_all.empty:
        print(f"- Train_all shape: {train_all.shape}")
    if not val_all.empty:
        print(f"- Val_all shape: {val_all.shape}")


if __name__ == "__main__":
    build_multiuser_datasets()
