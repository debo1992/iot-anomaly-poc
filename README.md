# IoT Smart Home — Anomaly Detection Proof-of-Concept

📌 Overview

This repository contains a full anomaly detection pipeline for smart homes. It simulates:
- Realistic multi-sensor data with injects anomalies: Refer **Iter 2 - Sensors**
- Trains multiple deep learning architectures: Refer **Justification of Anomaly Detection Models**
  - 🧠 **LSTM** — sequential modeling, baseline  
  - ⚡ **CNN** — 1D convolution with dilations for long context  
  - ⏱️ **TCN** — temporal convolutional network with residuals  
  - 🎯 **Transformer** — attention-based sequence encoder  
- Prepares models for deployment on edge devices: Refer **Deployment Prep & Productionizing the System**



## Repo Stucture
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
```

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
## Justification of Anomaly Detection Models
 I selected four deep learning architectures—LSTM, CNN, TCN, and Transformer—each offering distinct strengths and trade-offs. My goal was to balance model accuracy, temporal pattern learning, and efficiency for edge deployment.

| Model           | Accuracy on Long Sequences | Inference Speed  | Model Size | Suitability for Edge Deployment                  |
| --------------- | -------------------------- | ---------------- | ---------- | ------------------------------------------------ |
| **LSTM**        | High                       | Slow             | Large      | Baseline, less ideal due to sequential compute   |
| **CNN**         | Moderate to High           | Fast             | Small      | Excellent balance, highly suitable for edge      |
| **TCN**         | High                       | Fast             | Medium     | Strong edge candidate, balances depth & speed    |
| **Transformer** | Very High                  | Moderate to Slow | Large      | Powerful but heavy, better for cloud or gateways |

I initially implemented an LSTM baseline model, along with other baseline models using CNN and Transformer architectures. Among these, the LSTM outperformed the others in terms of accuracy. However, since CNNs are generally more edge-device friendly due to their efficiency and lower computational requirements, I decided to focus on improving the CNN model.

To do this, I studied the data patterns closely and made targeted adjustments — specifically, I tweaked the number of CNN layers and optimized the window size of input data the CNN could process. These modifications allowed the CNN to capture relevant temporal features more effectively, ultimately enabling it to outperform the LSTM model.

Following this, I quantized the CNN model. Quantization was especially suitable here because it could be applied easily to the entire CNN architecture, unlike the LSTM, where only the linear layers could be quantized effectively. This made the CNN model not only more efficient but also better suited for deployment within the tight time constraints of the project. 

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

## Productionizing the System

I’ve already developed quantization and benchmarking scripts, which streamline optimizing the CNN model for efficient deployment on edge devices. This makes real-time, low-latency inference feasible directly on-device.

For serving the model, tools like TensorFlow Lite or ONNX Runtime are ideal for running quantized models efficiently on edge hardware. If cloud-based inference is preferred, frameworks like TensorFlow Serving or TorchServe can handle scalable real-time prediction APIs.

To monitor data drift and model performance, I implemented a data drift monitoring system. For this, tools like Evidently AI or WhyLabs provide out-of-the-box capabilities for drift detection and metric tracking. Additionally, logging infrastructure can be built with Prometheus and visualized via Grafana dashboards for real-time alerts on model metrics and system health.

For retraining, a robust MLOps pipeline can be built using platforms such as Kubeflow, MLflow, or AWS SageMaker Pipelines. These tools help automate data ingestion, model retraining, validation, and deployment workflows. Retraining can be scheduled regularly or triggered by drift detection alerts to ensure the model adapts to new data distributions.

Together, these tools create an end-to-end system that supports efficient, scalable serving, continuous monitoring, and seamless retraining to maintain model accuracy and reliability in production.

---

## 🚀 Roadmap
- ✅ **Synthetic dataset generation** + anomaly injection  
- ✅ **Multi-model training & benchmarking**  
- ✅ **Quantization & ONNX conversion**  
- ✅ **Drift monitoring**  
- 🔜 **Deploy REST API** for smart home integration  
- 🔜 **Pilot with real IoT data**

## 🚀 USAGE

### 🧱 Instructions for Setting Up Environment (Setup)

```bash
# (1) Create and activate a virtual environment (optional)
python3 -m venv venv_name
source venv_name/bin/activate  

# (2) Install required packages
pip install -r requirements.txt
```

### 📦 Build Dataset

Before training, build the dataset:

```text
python3 main.py --build-data
```

### Training Default config:
```
python3 main.py
```


To run the model with the stored config
```
python3 main.py --config configs/main_train_config.json
```
If you want to override some configs, modify the example below:
```
python3 main.py --config configs/train_config.json --epochs 5 --model_type LSTM
```

```
| Argument            | Type  | Default     | Description                                 |
| ------------------- | ----- | ----------- | ------------------------------------------- |
| `--build-data`      | flag  | off         | Build the dataset from scratch              |
| `--config`          | str   | None        | Path to a JSON config file                  |
| `--model_type`      | str   | LSTM        | Model to train (`LSTM`, `CNN`, `TCN`, etc.) |
| `--epochs`          | int   | 10          | Number of training epochs                   |
| `--batch_size`      | int   | 64          | Batch size                                  |
| `--lr`              | float | 0.001       | Learning rate                               |
| `--patience`        | int   | 5           | Early stopping patience                     |
| `--window_size`     | int   | 32          | Time-series window size                     |
| `--loss_type`       | str   | weighted_ce | Type of loss function (`weighted_ce`, etc.) |
| `--balanced_loader` | bool  | False       | Use class-balanced data loading             |
| `--no-log`          | flag  | off         | Disable MLflow logging (dont use right now) |
```

### Quantize Model
Create the quantized model with PQT based on model type. Eport the model to onnx and save models.
```bash
python3 quantize_model.py
```

### Benchmark tests
```
python3 test/benchmark_compare.py
```



