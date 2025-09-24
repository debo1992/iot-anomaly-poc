import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_synthetic_data(
    start_date="2025-01-01",
    days=3,
    freq="5min",
    seed=42,
    output_dir="outputs/data",
    temp_base=21,
    humid_base=45,
    fridge_base=150
):
    """
    Generate synthetic IoT data with injected anomalies and priority-based labeling.

    Args:
        start_date (str): Start date of data in 'YYYY-MM-DD' format.
        days (int): Number of days to simulate.
        freq (str): Sampling frequency, e.g., '5min'.
        seed (int): Random seed for reproducibility.
        output_dir (str): Directory to save the CSV.
        temp_base (float): Base room temperature.
        humid_base (float): Base humidity level.
        fridge_base (float): Base fridge power consumption.

    Returns:
        df (pd.DataFrame): Synthetic dataset with anomaly class labels.
        anomalies (list): List of injected anomalies with timestamps & descriptions.
    """
    np.random.seed(seed)

    # ---------------------
    # Generate timestamps
    # ---------------------
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=days)
    timestamps = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    n = len(timestamps)

    # ---------------------
    # Base Signals
    # ---------------------
    temp_signal = temp_base + 3 * np.sin(
        2 * np.pi * (timestamps.hour + timestamps.minute/60) / 24 + 5*np.pi/4
    )
    temperature = np.array(temp_signal + np.random.normal(0, 0.5, n))

    humidity = humid_base + np.random.normal(0, 2, n)
    shower_mask = ((timestamps.hour == 7) | (timestamps.hour == 19))
    humidity[shower_mask] += np.random.uniform(20, 30, shower_mask.sum())

    fridge = fridge_base + 10*np.sin(np.linspace(0, 50*np.pi, n)) + np.random.normal(0, 5, n)

    door = np.zeros(n)
    for hour in [8, 18]:
        door[(timestamps.hour == hour) & (timestamps.minute < 10)] = 1

    fire_alarm = np.zeros(n)

    # ---------------------
    # Anomaly storage
    # ---------------------
    anomalies = []
    class_labels = np.zeros(n, dtype=int)  # 0 = normal, 1–5 = anomaly type

    # Helper: assign anomaly with priority
    def assign_anomaly(idx, label, desc):
        nonlocal class_labels
        if label > class_labels[idx]:
            class_labels[idx] = label
            anomalies.append((timestamps[idx], desc))

    # ---------------------
    # Injected Anomalies
    # ---------------------

    # A1: Temperature drop → class 1
    drop_start = np.random.randint(0, n - 12)
    temperature[drop_start:drop_start+12] -= 5
    for i in range(drop_start, drop_start+12):
        assign_anomaly(i, 1, "Temperature drop (heating failure)")

    # A2: Humidity spike → class 2
    non_shower_idx = np.where(~shower_mask)[0]
    idx = np.random.choice(non_shower_idx)
    humidity[idx-2:idx] += 40
    assign_anomaly(idx, 2, "Unexpected bathroom humidity spike")

    # A3: Fridge power failure → class 3
    fail_start = np.random.randint(0, n - 24)
    fridge[fail_start:fail_start+24] = 0
    for i in range(fail_start, fail_start+24):
        assign_anomaly(i, 3, "Fridge power failure (2h outage)")

    # A4: Door opened at night → class 4
    night_indices = np.where((timestamps.hour >= 1) & (timestamps.hour <= 3))[0]
    if len(night_indices) > 0:
        idx = np.random.choice(night_indices)
        door[idx] = 1
        assign_anomaly(idx, 4, "Front door opened at night")

    # A5: Fire alarm triggered → class 5
    alarm_idx = np.random.randint(0, n - 3)
    fire_alarm[alarm_idx:alarm_idx+3] = 1
    for i in range(alarm_idx, alarm_idx+3):
        assign_anomaly(i, 5, "Fire alarm triggered")

    # ---------------------
    # Final DataFrame
    # ---------------------
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "fridge_power_w": np.round(fridge, 1),
        "front_door_open": door.astype(int),
        "fire_alarm": fire_alarm.astype(int),
        "anomaly_class": class_labels  # 0 = normal, 1–5 anomaly
    })

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "synthetic_iot_data.csv")
    df.to_csv(filepath, index=False)

    return df, anomalies


if __name__ == "__main__":
    df, anomalies = generate_synthetic_data()
    print("Generated dataset:", df.shape)
    print("\nInjected anomalies (priority applied):")
    for ts, desc in anomalies:
        print(f"- {ts}: {desc}")
    print("\nSample data:\n", df.head())
