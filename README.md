# IoT Smart Home — Anomaly Detection Proof-of-Concept

```text
iot-anomaly-poc/
├─ README.md
├─ requirements.txt
├─ generate_data.py
├─ detect_anomalies.py
└─ outputs/           # Stores generated datasets, logs, anomaly reports, plots

## Overview
This repository contains a complete proof-of-concept for anomaly detection on multi-sensor smart-home time-series data. It simulates sensors, injects anomalies, and runs a lightweight detection pipeline that uses both interpretable rules and an unsupervised multivariate model.

## Sensors Simulated

- **Living room temperature (°C)**  
  Normal daily cycle: ~20–24°C with small noise.  
  *Anomaly*: sudden drops to simulate heating failure.

- **Bathroom humidity (%)**  
  Baseline ~45%. Spikes during shower times (7–8am, 7–8pm).  
  *Anomaly*: sudden spikes outside shower hours.

- **Fridge power usage (Watts)**  
  Normal ~150W with small fluctuations.  
  *Anomaly*: power failure (flatlined at 0W).

- **Hallway motion (binary)**  
  More likely during day hours (7am–11pm).  

- **Front door (binary)**  
  Typically opened around 8am (leaving) and 6pm (returning).  
  *Anomaly*: unexpected night-time door opening.

- **Fire alarm (binary)**  
  Normally off.  
  *Anomaly*: triggered unexpectedly (critical event).

## Anomalies Injected

1. **Front door opened at night** – suspicious behavior between 1–3am.  
2. **Fridge power failure** – fridge power flatlines at `0W` for ~2 hours.  
3. **Unexpected humidity spike** – abnormal rise outside normal shower times.  
4. **Fire alarm triggered** – safety-critical anomaly lasting ~15 minutes.  
5. **Temperature drop** – simulated heating failure with a sudden 5°C drop.

## Dataset

- Frequency: **5-minute samples**  
- Duration: configurable (default = 3 days)  
- Output: `outputs/data/synthetic_iot_data.csv`  

Each row contains:  

| timestamp           | temperature_c | humidity_pct | fridge_power_w | front_door_open | fire_alarm |
|---------------------|---------------|--------------|----------------|-----------------|------------|
| 2025-01-01 00:00:00 | 21.3          | 45.2         | 150.1          | 0               | 0          |

## Usage

Generate synthetic data:
```bash
python generate_data.py