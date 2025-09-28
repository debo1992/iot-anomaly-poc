# IoT Smart Home — Anomaly Detection Proof-of-Concept

📌 Overview

This repository contains a full anomaly detection pipeline for smart homes. It simulates realistic multi-sensor data, injects anomalies, trains multiple deep learning architectures, and prepares them for deployment on edge devices.

The goal: move beyond brittle threshold rules and deliver adaptive, learning-based monitoring for safety, efficiency, and security.


```text
iot-anomaly-poc/
├── main.py                 # Training pipeline (multi-model)
├── build_multiuser_datasets.py  # Generate multi-user datasets
├── datasets/               # Synthetic IoT datasets + generator
│   ├── anomaly_dataset.py  # Custom PyTorch dataset + sampler
│   ├── generate_data.py    # Sensor simulation + anomaly injection
│   ├── data/               # Train/val CSVs
│   └── README_datasets.md
├── models/                 # Baseline deep learning models
│   ├── lstm_basic.py
│   ├── cnn_basic.py
│   ├── tcn_basic.py
│   ├── transformer_basic.py
│   └── initialize_model.py
├── utils/                  # Training utilities
│   ├── evaluation_metrics.py  # Confusion matrix, PR curves, F1
│   ├── logging.py             # MLflow + W&B logging
│   ├── class_weight.py        # Handle class imbalance
│   └── losses.py              # Focal loss, weighted CE
├── tests/                 # Post-training tools
│   ├── load_eval_model.py     # Baseline model evaluation
│   ├── quantize_model.py      # Dynamic & static quantization 
│   ├── drift_monitor.py       # Data drift detection
│   └── benchmark_compare.py   # Compare model sizes + ONNX
├── outputs/               # Saved models
├── requirements.txt
└── README.md


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

## 📊 Dataset Organization
- **`train_users/`** → 80 simulated households (6 months)  
- **`val_users/`** → 20 households  
- **`train_all.csv`, `val_all.csv`** → aggregated datasets  

Includes **seasonality** (winter/summer drift) and **random anomaly durations** (2 hours – 1 week) for realism.  

---

## 🤖 Models
Implemented baselines:
- 🧠 **LSTM** — sequential modeling, baseline  
- ⚡ **CNN** — 1D convolution with dilations for long context  
- ⏱️ **TCN** — temporal convolutional network with residuals  
- 🎯 **Transformer** — attention-based sequence encoder  

---

## 🧪 Training Pipeline
- 🎲 **Weighted sampling** for class imbalance  
- 📉 **Custom losses**: CrossEntropy, Focal Loss  
- ⏳ **Early stopping** + best model checkpointing  
- 📊 **Experiment tracking** via MLflow (confusion matrices & PR curves)  
- 🟣 **W&B optional logging**  

---

## 📦 Deployment Prep
- 📏 **Quantization** (PyTorch dynamic/static: CNN, TCN, LSTM)  
- 🔄 **ONNX export** for cross-platform inference  
- ⚖️ **Model size benchmarking** (original vs quantized vs ONNX)  
- 📡 **Drift monitoring** for household behavior changes  
- 🐳 **Docker-ready packaging**  

---

## 🚀 Roadmap
- ✅ **Synthetic dataset generation** + anomaly injection  
- ✅ **Multi-model training & benchmarking**  
- ✅ **Quantization & ONNX conversion**  
- ✅ **Drift monitoring**  
- 🔜 **Deploy REST API** for smart home integration  
- 🔜 **Pilot with real IoT data**  


