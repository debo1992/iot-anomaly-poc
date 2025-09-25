# IoT Smart Home — Anomaly Detection Proof-of-Concept

```text
iot-anomaly-poc/
├── README.md
├── __pycache__
│   ├── generate_data.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── build_multiuser_datasets.py
├── datasets
│   ├── README_datasets.md
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── anomaly_dataset.cpython-310.pyc
│   │   └── generate_data.cpython-310.pyc
│   ├── anomaly_dataset.py
│   ├── data
│   │   ├── train
│   │   ├── train_all.csv
│   │   ├── v1
│   │   │   ├── train
│   │   │   ├── train_all.csv
│   │   │   ├── val
│   │   │   └── val_all.csv
│   │   ├── val
│   │   └── val_all.csv
│   └── generate_data.py
├── main.py
├── models
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── lstm_basic.cpython-310.pyc
│   └── lstm_basic.py
├── plot_datasetv2.png
├── requirements.txt
└── utils
    ├── __pycache__
    │   └── plot_iot_data.cpython-310.pyc
    └── plot_iot_data.py

## Overview
This repository contains a complete proof-of-concept for anomaly detection on multi-sensor smart-home time-series data. It simulates sensors, injects anomalies, and runs a lightweight detection pipeline that uses both interpretable rules and an unsupervised multivariate model.

# IoT Anomaly Detection POC

## Iter 1 - Sensors: Baic dataset with anomalies but no drift simulation

## Iter 2 - Sensors
- **Temperature (°C, Living Room)**  
  - Baseline 21 °C ± daily cycle, with slow drift (+0.05 °C/week).  
  - Anomalies: sensor failure (constant/frozen values).  

- **Humidity (%RH, Bathroom)**  
  - Baseline 45 %RH, spikes to 70–90 % during showers.  
  - Drift: +0.1 %RH/week.  
  - Anomalies: spikes outside shower schedule.  

- **Fridge Power (W)**  
  - Baseline ~150 W, with compressor cycling ±10 W.  
  - Anomalies: power failure (drop to 0 W).  

- **Front Door (binary)**  
  - 0 = closed, 1 = open.  
  - Anomalies: opening during 00:00–05:00 (nighttime).  

- **Fire Alarm (binary)**  
  - 0 = off, 1 = alarm triggered.  
  - Overrides all anomalies.  

## Labels
Each timestamp has a label:  
- 0 → Normal  
- 1 → Temperature anomaly  
- 2 → Humidity anomaly  
- 3 → Fridge anomaly  
- 4 → Door anomaly  
- 5 → Fire alarm (highest priority)  

## Dataset Organization
- `train_users/` → 80 users, hourly data over 6 months.  
- `val_users/` → 20 users.  
- `train_all.csv`, `val_all.csv` → concatenated datasets.  

## Limitations
- Synthetic dataset: not based on real hardware logs.  
- Drift patterns are modeled linearly, while real drift can be nonlinear or environment-dependent.  
- Event frequencies are approximated; actual user behavior varies.  
- Rare anomalies (like fire alarms) are injected more frequently than real-world rates for training utility.  

Weather Seasonality - in addition to sensor drift:

January (cold + dry): Temperature −5 °C, Humidity −10% RH.

April (hot + humid): Temperature +5 °C, Humidity +10% RH.

February–March: Linearly interpolated between these extremes.

Other months remain neutral for this POC.

## Next Steps
- Train anomaly classifiers (LSTM baseline included).  
- Explore domain adaptation for real IoT datasets.  



## Usage

Generate synthetic data:
```bash
python generate_data.py

dataset version 2 randomised the duration and keeping it to one event per catastropy
also randomised the heat and humidity anomaly levels

assumption is that there is a priority 5>4>3>2>1
take 2hr - 1 week to repair aircon heater

Future:
increase number of events
softmax to predict soft probabilities to detect overlapping events rather than hard prioritization

change plot x-axis in terms of number of hours to observe aircon failure duration

