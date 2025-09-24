# IoT Smart Home — Anomaly Detection Proof-of-Concept

### Folder Structure 
iot-anomaly-poc/
├─ README.md              # Project overview, setup, usage instructions
├─ requirements.txt       # Python venv dependencies
├─ generate_data.py       # Script to simulate/generate IoT sensor data
├─ detect_anomalies.py    # Script to detect anomalies in the generated data
└─ outputs/               # Stores generated datasets, logs, anomaly reports, plots

## Overview
This repository contains a complete proof-of-concept for anomaly detection on multi-sensor smart-home time-series data. It simulates sensors, injects anomalies, and runs a lightweight detection pipeline that uses both interpretable rules and an unsupervised multivariate model.

