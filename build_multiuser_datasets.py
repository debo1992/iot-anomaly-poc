import os
import numpy as np
import pandas as pd
from datasets.generate_data import generate_synthetic_data
from utils.plot_iot_data import plot_iot_data

def build_multiuser_datasets(
    train_users=80,
    val_users=20,
    start_date="2025-01-01",
    days=3,
    freq="5min",
    seed=123,
    output_dir="datasets/data"
):
    np.random.seed(seed)

    # Directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_dfs = []
    val_dfs = []

    # Helper to create per-user dataset
    def make_user_dataset(user_id, split_dir, store_list):
        # Random base values with jitter
        temp_base = 21 + np.random.uniform(-2, 2)
        humid_base = 45 + np.random.uniform(-5, 5)
        fridge_base = 150 + np.random.uniform(-20, 20)

        df, _ = generate_synthetic_data(
            start_date=start_date,
            days=days,
            freq=freq,
            temp_base=temp_base,
            humid_base=humid_base,
            fridge_base=fridge_base,
            seed=np.random.randint(0, 10000),
            output_dir=split_dir,
        )

        # Add user_id column
        df["user_id"] = user_id
        # plot_iot_data(df, user_id=user_id, figsize=(15, 12), save_path="plot.png")
        # Save per-user
        filepath = os.path.join(split_dir, f"user_{user_id}.csv")
        df.to_csv(filepath, index=False)

        store_list.append(df)

    # Build train users
    for uid in range(1, train_users + 1):
        make_user_dataset(uid, train_dir, train_dfs)

    # Build val users
    for uid in range(train_users + 1, train_users + val_users + 1):
        make_user_dataset(uid, val_dir, val_dfs)

    # Save combined datasets
    train_all = pd.concat(train_dfs, ignore_index=True)
    val_all = pd.concat(val_dfs, ignore_index=True)

    train_all.to_csv(os.path.join(output_dir, "train_all.csv"), index=False)
    val_all.to_csv(os.path.join(output_dir, "val_all.csv"), index=False)

    print(f"✅ Generated {train_users} train users and {val_users} val users")
    print(f"Train dataset shape: {train_all.shape}")
    print(f"Val dataset shape:   {val_all.shape}")


if __name__ == "__main__":
    build_multiuser_datasets()
    # Example: generate 50 train users and 10 val users