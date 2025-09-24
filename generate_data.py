import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_synthetic_data(
    start_date="2025-01-01",
    days=3,
    freq="5min",
    seed=42,
    output_dir="outputs/data"
):
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

    # 1. Living room temperature (°C)
    # Daily cycle: cooler at night, warmer mid-day
    temp_base = 21 + 3 * np.sin(2 * np.pi * (timestamps.hour + timestamps.minute/60) / 24)
    temperature = temp_base + np.random.normal(0, 0.5, n)

    # 2. Bathroom humidity (%)
    humidity = 45 + np.random.normal(0, 2, n)
    # Add shower spikes (7–8am, 7–8pm typical times)
    shower_mask = ((timestamps.hour == 7) | (timestamps.hour == 19))
    humidity[shower_mask] += np.random.uniform(20, 30, shower_mask.sum())

    # 3. Fridge power consumption (Watts)
    # Runs cycles: ~150W with small fluctuations
    fridge = 150 + 10*np.sin(np.linspace(0, 50*np.pi, n)) + np.random.normal(0, 5, n)


    # 4. Front door (binary open/close)
    door = np.zeros(n)
    # Normal door usage: morning (8am) + evening (6pm)
    for hour in [8, 18]:
        door[(timestamps.hour == hour) & (timestamps.minute < 10)] = 1

    # 5. Fire alarm (binary, mostly off)
    fire_alarm = np.zeros(n)

    # ---------------------
    # Injected Anomalies
    # ---------------------
    anomalies = []

    # A1: Door opened at night (1–3am)
    night_indices = np.where((timestamps.hour >= 1) & (timestamps.hour <= 3))[0]
    if len(night_indices) > 0:
        idx = np.random.choice(night_indices)
        door[idx] = 1
        anomalies.append((timestamps[idx], "Front door opened at night"))

    # A2: Fridge power failure (flat at 0W for ~2 hours)
    fail_start = np.random.randint(0, n - 24)
    fridge[fail_start:fail_start+24] = 0
    anomalies.append((timestamps[fail_start], "Fridge power failure (2h outage)"))

    # A3: Unexpected humidity spike (outside shower hours)
    non_shower_idx = np.where(~shower_mask)[0]
    idx = np.random.choice(non_shower_idx)
    humidity[idx] += 40
    anomalies.append((timestamps[idx], "Unexpected bathroom humidity spike"))

    # A4: Fire alarm triggered (random rare event)
    alarm_idx = np.random.randint(0, n)
    fire_alarm[alarm_idx:alarm_idx+3] = 1  # stays on for ~15 minutes
    anomalies.append((timestamps[alarm_idx], "Fire alarm triggered"))

    # A5: Sudden drop in temperature (heating failure at night)
    drop_start = np.random.randint(0, n - 12)
    temperature[drop_start:drop_start+12] -= 5
    anomalies.append((timestamps[drop_start], "Temperature drop (heating failure)"))

    # ---------------------
    # Build Dataset
    # ---------------------
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "fridge_power_w": np.round(fridge, 1),
        "front_door_open": door.astype(int),
        "fire_alarm": fire_alarm.astype(int)
    })

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "synthetic_iot_data.csv")
    df.to_csv(filepath, index=False)

    return df, anomalies


if __name__ == "__main__":
    df, anomalies = generate_synthetic_data()
    print("Generated dataset:", df.shape)
    print("\nInjected anomalies:")
    for ts, desc in anomalies:
        print(f"- {ts}: {desc}")
