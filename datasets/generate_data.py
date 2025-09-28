"""
File: generate_data.py
Description:
    This script contains functions and utilities for generating our projects sensor data of temperature,
    humidity, fridge power, door opens and fire alarm thats injected with anomalies. This data is generated only for one user. 
    This function is called by generate_multiple_users_data.py to generate data for multiple users.

Key Features:
- Generates synthetic datasets with configurable parameters.
- Simulates both normal and anomalous data patterns for anomaly detection.
- Supports exporting generated data to CSV.
- Includes utilities for visualizing the generated data.

# datasets/generate_data.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.plot_iot_data import plot_iot_data



def seasonal_adjustments(timestamps):
    """
    Vectorized seasonal adjustments (°C and %RH).
    - Jan: cold (-5°C) and dry (-10% RH)
    - Apr: hot (+5°C) and humid (+10% RH)
    - Feb/Mar: linear interpolation between Jan and Apr
    - Others: 0
    """
    ts = pd.Series(timestamps)
    months = ts.dt.month.values
    temp_adj = np.zeros(len(timestamps), dtype=float)
    humid_adj = np.zeros(len(timestamps), dtype=float)

    for i, m in enumerate(months):
        if m == 1:
            temp_adj[i] = -5.0
            humid_adj[i] = -10.0
        elif m == 4:
            temp_adj[i] = +5.0
            humid_adj[i] = +10.0
        elif m in (2, 3):
            factor = (m - 1) / 3.0  # 2 -> 1/3, 3 -> 2/3
            temp_adj[i] = -5.0 + factor * (5.0 - (-5.0))
            humid_adj[i] = -10.0 + factor * (10.0 - (-10.0))
        else:
            temp_adj[i] = 0.0
            humid_adj[i] = 0.0

    return temp_adj, humid_adj


def generate_user_data(
    user_id,
    start_date="2025-01-01",
    days=180,
    freq="1H",
    base_temp=21.0,
    base_temp_amp=2.0,
    base_humidity=45.0,
    fridge_base=150.0,
    temp_drift_per_week=0.05,     # °C per week
    humid_drift_per_week=0.10,    # %RH per week
    noise_temp=0.3,               # std dev °C
    noise_humid=2.0,              # std dev %RH
    noise_fridge=3.0,             # std dev W
    dropout_prob=0.001,           # fraction of samples that start a dropout
    dropout_mean_duration=3,      # mean dropout length in samples
    fridge_fail_hours=2,          # fridge failure duration (hours)
    fire_duration_hours=1,        # fire alarm duration (hours)
    seed=None
):
    """
    Generate a time-series DataFrame for a single user with realistic noise, drift, and labels.

    Returns:
        df (pd.DataFrame): columns -> ['timestamp','temperature_c','humidity_pct',
                                       'fridge_power_w','front_door_open','fire_alarm',
                                        'anomaly_class', plus metadata columns]
        anomalies (list): list of (timestamp, description) injected (for quick inspection)
    """
    if seed is not None:
        np.random.seed(seed)

    # timestamps
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(days=days)
    timestamps = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    n = len(timestamps)

    # samples -> hours per sample (to convert durations)
    hours_per_sample = pd.to_timedelta(freq).total_seconds() / 3600.0
    # weeks array (for drift)
    weeks = (np.arange(n) * hours_per_sample) / (24.0 * 7.0)

    # seasonal adjustments
    temp_season, humid_season = seasonal_adjustments(timestamps)

    # Temperature baseline (circadian daily cycle) + drift + season + noise
    circadian = base_temp_amp * np.sin(2.0 * np.pi * (timestamps.hour / 24.0)+ 5 * np.pi/4)
    temp_base_signal = base_temp + circadian
    temperature = np.array(
        temp_base_signal
        + weeks * temp_drift_per_week
        + temp_season
        + np.random.normal(0.0, noise_temp, n)
    )

    # Humidity baseline + drift + season + shower spikes + noise
    humidity = (
        base_humidity
        + weeks * humid_drift_per_week
        + humid_season
        + np.random.normal(0.0, noise_humid, n)
    )
    # shower spikes at typical times (can be more variable later)
    shower_mask = (timestamps.hour.isin([7, 19, 22]))
    humidity[shower_mask] += np.random.uniform(20.0, 30.0, shower_mask.sum())

    # Fridge power (Watts): base + compressor cyclic behaviour + noise
    # Compressor cycle simulated with multi-frequency sinusoids
    t_lin = np.linspace(0, 4 * np.pi, n)
    fridge_power = (
        fridge_base
        + 8.0 * np.sin(0.5 * t_lin)       # slow cycle
        + 4.0 * np.sin(6.0 * t_lin)       # faster oscillation
        + np.random.normal(0.0, noise_fridge, n)
    )

    # Door: deterministic opens at 8 & 18 plus occasional random opens
    door = np.zeros(n, dtype=int)
    door[(timestamps.hour == 8) | (timestamps.hour == 18)] = 1
    # small random openings during the day
    prob_day = 0.02
    prob_night = 0.002
    rand_probs = np.random.rand(n)
    day_mask = (timestamps.hour >= 7) & (timestamps.hour <= 22)
    door[(rand_probs < prob_day) & day_mask] = 1
    door[(rand_probs < prob_night) & (~day_mask)] = 1

    # Fire alarm baseline (rare)
    fire_alarm = np.zeros(n, dtype=int)

    # Metadata / anomaly label container
    anomaly_class = np.zeros(n, dtype=int)  # 0 normal, 1..5 anomalies
    anomalies = []

    # ---------- Sensor dropouts ----------
    # Random dropout starts; set a contiguous block to NaN then forward-fill later
    n_drop_starts = max(0, int(dropout_prob * n))
    if n_drop_starts > 0:
        drop_starts = np.random.choice(np.arange(n), size=n_drop_starts, replace=False)
        for ds in drop_starts:
            dur = max(1, int(np.random.poisson(dropout_mean_duration)))
            end = min(n, ds + dur)
            temperature[ds:end] = np.nan
            humidity[ds:end] = np.nan
            fridge_power[ds:end] = np.nan
            # we won't mark these as anomaly_class but will set a 'dropout' flag later if needed

    # ---------- Inject anomalies (with priority logic) ----------
    def mark_range_max(label, start_idx, length):
        """Mark range [start_idx, start_idx+length) with label applying priority (max)."""
        end_idx = min(n, start_idx + length)
        nonlocal anomaly_class
        # only upgrade label where label > existing
        anomaly_class[start_idx:end_idx] = np.maximum(anomaly_class[start_idx:end_idx], label)

    # 1) Temperature drop (heating failure) — multi-sample
    anomaly_duration = np.random.randint(2, 168)
    temp_drop_len = max(1, int(round(anomaly_duration / hours_per_sample)))  # default 12 samples scaled by sample rate
    temp_drop_start = np.random.randint(0, max(1, n - temp_drop_len))
    temperature[temp_drop_start:temp_drop_start + temp_drop_len] -= np.random.randint(2, 12)
    mark_range_max(1, temp_drop_start, temp_drop_len)
    anomalies.append((timestamps[temp_drop_start], "Temperature drop (heating failure)"))

    # 2) Humidity spike outside shower hours
    non_shower_idxs = np.where(~shower_mask)[0]
    if len(non_shower_idxs) > 0:
        hs_idx = np.random.choice(non_shower_idxs)
        humid_spike_len = max(1, int(round(3.0 / hours_per_sample)))
        humidity[hs_idx:hs_idx + humid_spike_len] += np.random.randint(30, 60)
        mark_range_max(2, hs_idx, humid_spike_len)
        anomalies.append((timestamps[hs_idx], "Unexpected humidity spike"))

    # 3) Fridge power failure — set to 0 for default fridge_fail_hours
    fridge_fail_samples = max(1, int(round(fridge_fail_hours / hours_per_sample)))
    fridge_fail_start = np.random.randint(0, max(1, n - fridge_fail_samples))
    fridge_power[fridge_fail_start:fridge_fail_start + fridge_fail_samples] = 0.0
    mark_range_max(3, fridge_fail_start, fridge_fail_samples)
    # correlated effect: small rise in temperature during and after outage
    temp_rise_len = min(n, fridge_fail_samples + max(1, int(round(np.random.randint(2,24) / hours_per_sample))))
    temperature[fridge_fail_start:fridge_fail_start + temp_rise_len] += np.linspace(0.2, 1.0, temp_rise_len)
    anomalies.append((timestamps[fridge_fail_start], "Fridge power failure"))

    # 4) Door opened at night (explicit suspicious open)
    night_idxs = np.where((timestamps.hour >= 1) & (timestamps.hour <= 3))[0]
    if len(night_idxs) > 0:
        door_idx = np.random.choice(night_idxs)
        door[door_idx] = 1
        mark_range_max(4, door_idx, 1)
        anomalies.append((timestamps[door_idx], "Front door opened at night"))

    # 5) Fire alarm triggered (rare) — duration measured in hours
    fire_samples = max(1, int(round(fire_duration_hours / hours_per_sample)))
    fire_start = np.random.randint(0, max(1, n - fire_samples))
    fire_alarm[fire_start:fire_start + fire_samples] = 1
    # fire overrides other labels (priority 5)
    anomaly_class[fire_start:fire_start + fire_samples] = 5
    # correlated effects: temp & humidity spike + noisy behavior + fridge disturbances
    for i in range(fire_start, min(n, fire_start + fire_samples)):
        temperature[i] = temperature[i] + np.random.uniform(5.0, 12.0)
        humidity[i] = humidity[i] + np.random.uniform(10.0, 25.0)
        fridge_power[i] = fridge_power[i] + np.random.uniform(-20.0, 20.0)
    anomalies.append((timestamps[fire_start], "Fire alarm triggered"))

    # ---------- Handle sensor dropouts (forward-fill for trainability) ----------
    # Keep a dropout flag column if desired
    # Convert arrays to pandas Series for ffill convenience
    temp_s = pd.Series(temperature)
    hum_s = pd.Series(humidity)
    fridge_s = pd.Series(fridge_power)

    # forward-fill, then backfill any leading NaNs with baseline values
    temp_s.ffill(inplace=True)
    temp_s.bfill(inplace=True)
    hum_s.ffill(inplace=True)
    hum_s.bfill(inplace=True)
    fridge_s.ffill(inplace=True)
    fridge_s.bfill(inplace=True)

    # convert back
    temperature = temp_s.values
    humidity = hum_s.values
    fridge_power = fridge_s.values

    # ---------- Compose final DataFrame ----------
    df = pd.DataFrame({
        "timestamp": timestamps,
        "hours": timestamps.hour,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "fridge_power_w": np.round(fridge_power, 1),
        "front_door_open": door.astype(int),
        "fire_alarm": fire_alarm.astype(int),
        "anomaly_class": anomaly_class.astype(int)
    })

    # add per-user metadata columns (constant across rows)
    metadata = {
    "user_id": str(user_id),
    "meta_base_temp": base_temp,
    "meta_base_humidity": base_humidity,
    "meta_fridge_base": fridge_base,
    "meta_temp_drift_per_week": temp_drift_per_week,
    "meta_humid_drift_per_week": humid_drift_per_week,
    "meta_noise_temp": noise_temp,
    "meta_noise_humid": noise_humid,
    "meta_noise_fridge": noise_fridge,
    "meta_dropout_prob": dropout_prob,
    "anomalies": anomalies
}
    # plot_iot_data(df, user_id=None, figsize=(15, 12), save_path="plot.png")

    return df, metadata
