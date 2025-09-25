# Synthetic IoT Data Documentation

## Overview

The synthetic IoT dataset is generated to simulate real-world smart home environments with IoT devices that collect data such as temperature, humidity, fridge power consumption, door states, and fire alarm status. The data contains several anomalies (e.g., temperature drops, humidity spikes, fridge failures) for testing and model validation.

## Folder Structure

The data generation script produces the dataset and can optionally save the data to disk in CSV format. Here's the general folder structure:

```
datasets/
│
├── data/
│   ├── train/                # Individual CSVs for each training user
│   │   ├── user_1.csv
│   │   ├── user_2.csv
│   │   └── ...
│   ├── val/                  # Individual CSVs for each validation user
│   │   ├── user_81.csv
│   │   └── ...
│   ├── train_all.csv         # Combined dataset for all train users
│   └── val_all.csv           # Combined dataset for all val users
│
├── datasets/
│   ├── generate_data.py      # Synthetic data generator
│   └── build_multiuser_datasets.py
│
├── utils/
│   └── plot_iot_data.py      # Visualization utilities
│
└── README.md

## Key Variables in one CSV

| timestamp           | temperature_c | humidity_pct | fridge_power_w | front_door_open | fire_alarm | fire_alarm |
|---------------------|---------------|--------------|----------------|-----------------|------------|------------|
| 2025-01-01 00:00:00 | 21.3          | 45.2         | 150.1          | 0               | 0          | 0-5        |

The dataset includes several variables (features) that represent environmental conditions, device states, and anomalies (labels).

---

## 🏷️ Label Classes

Each sample is assigned one **class label**:

| Class | Anomaly Type              | Priority |
|-------|---------------------------|----------|
| 0     | ✅ Normal                 | –        |
| 1     | 🌡️ Temp anomaly          | Low      |
| 2     | 💧 Humidity anomaly       | Medium   |
| 3     | ⚡ Fridge anomaly         | High     |
| 4     | 🚪 Door anomaly           | Higher   |
| 5     | 🔔 Fire alarm             | Highest  |

⚠️ **If multiple anomalies occur at the same timestamp:**  
Priority is enforced as: **5 > 4 > 3 > 2 > 1**.  

---

## 🚀 Usage

### Generate Multiuser Dataset
```bash
python datasets/build_multiuser_datasets.py


---

## 📊 Dataset Versions

### 🔹 **Version 1 (V1) – Basic Synthetic Dataset**
- **Sensors**:  
  | Sensor          | Unit  | Notes |
  |-----------------|-------|-------|
  | 🌡️ Temperature | °C    | Living room temp |
  | 💧 Humidity     | %     | Bathroom humidity (spikes after showers) |
  | ⚡ Fridge Power | W     | Normal baseline usage with failures injected |
  | 🚪 Door State   | 0/1   | Front door open/close events |
  | 🔔 Fire Alarm   | 0/1   | Alarm events |

- **Anomalies injected**:
  - 🌡️ Unusual temperature drops/spikes
  - 💧 Bathroom humidity spikes (showers)
  - ⚡ Fridge power failures
  - 🚪 Front door opening at unusual hours
  - 🔔 Fire alarm triggered (rare, high-priority)

- **Limitations**:
  - Signals mostly periodic  
  - No long-term drift or seasonal patterns  
  - Fixed duration of anomalies
  - One occurance
  - Fixed magnitude of temp, humidty drop etc
  - no correlation of variables

    ## 📌 Assumptions

### 🔹 Normal Behavior
Environmental conditions (temperature, humidity, fridge power, door states) follow predictable patterns that mimic real-world behavior:

- 🌡️ **Temperature** → follows a daily cycle based on sinusoidal oscillation.  
- 💧 **Humidity** → increases during showering hours (**7:00 AM** and **7:00 PM**).  
- ⚡ **Fridge Power** → cyclical behavior with added noise.  
- 🚪 **Door States** → mostly closed, with occasional open events.

---

### 🔹 Anomalies
The injected anomalies are designed to reflect common faults or unusual events in a smart home environment:  
- Each anomaly is **randomly sampled** (time, duration, severity).  
- Ensures **variability** in dataset runs.  

---

### 🔹 Timeframe
- Default simulation period: **3 days**  
- Configurable **start dates**  
- Adjustable **duration**  

---

### 🔹 Sampling Frequency
- Configurable sampling rate  
- Supported frequencies:  
  - ⏱️ **5 minutes**  
  - ⏱️ **1 hour**  
  - (or user-defined)

---

## 📌 Version 2 (V2) – Enhanced Synthetic Dataset

### 🔹 New Features
- 📈 **Linear sensor drift** over months (e.g., thermal camera bias)  
- 🌦️ **Seasonal variation**:  
  - ❄️ January → colder & drier  
  - ☀️ April → hotter & more humid  
- 🔗 **Correlation between parameters**:  
  - 🔥 Fire alarm linked with sudden temperature spike  
  - 🚪 Door activity tied to motion & fridge usage  
- 📊 **More realistic noise models** (thermal + environmental fluctuations)

---

### 🔹 Anomalies Injected
(Same as V1 + new correlations)
- 🌡️ Temperature drift or faults  
- 💧 Humidity spikes out of seasonal range  
- ⚡ Fridge abnormal usage or failures  
- 🚪 Door anomalies at unexpected times  
- 🔔 Fire alarm (critical, with correlated temp rise)  

---

### 🔹 Key Enhancements in V2
1. **Dynamic Anomaly Duration**  
   - Randomized from **2 hours → 1 week**  
   - Simulates repair time variability  
   - *Example*: Heating failure may last hours or persist for several days  

2. **Randomized Anomaly Intensity**  
   - Varying magnitudes for anomalies  
   - *Example*: Temperature drop could be –5 °C or –10 °C depending on fault severity  

3. **Priority-Based Labeling**  
   - Fire (5) > Door (4) > Fridge (3) > Humidity (2) > Temperature (1)  
   - Ensures most critical anomaly is labeled when overlaps occur  

4. **Seasonal Variations**  
   - Winter (colder, drier) vs Summer (hotter, humid)  
   - Impacts baseline signals and anomalies  

5. **Realistic Repair Times**  
   - Devices (AC, heaters, fridges) take **2h–1w** to recover  
   - Better simulates real maintenance delays  

6. **Realistic Noise Models**  
   - Sensor drift + wear simulated  
   - Variance grows with age or environmental stress  
   - *Example*: Humidity sensors show noisier signals in high humidity  

---

### 🔹 Limitations
- ⚙️ Still **rule-based**, not live IoT feeds  
- 📉 Drift modeled as **linear** (real drift often nonlinear)  
- 🌦️ Seasonal model simplified (only temp & humidity affected)  
- ⏳ Limited coverage → mostly short-term data windows  
- 📊 Noise models remain simplistic vs real-world sensor physics  
- 🛠️ Repairs modeled as full recovery (no partial or cascading failures)  

---

### 📌 Assumptions in V2
- Each **user** = separate zone with unique sensor baselines  
- All users share **seasonal trends**, but differ in baseline preferences + variances  
- Anomalies injected **independently**, but priority rules ensure correct class labeling  
- Seasonal + noise effects applied **globally**, not individually per user behavior  

---

### ✅ Summary
V2 provides:
- More **realistic anomaly simulation**  
- Better **temporal dynamics** (duration, intensity, repair)  
- **Correlated signals** for multi-sensor fault realism  
- Seasonal + drift effects for **longer-term realism**  

But still limited by simplified rules, short temporal scope, and lack of complex inter-user behaviors.

---

## 🚀 Next Steps

✅ **V3 – User Behavior Models**  
- Simulate realistic household routines  
- Examples: varying shower times, fridge open/close cycles, irregular door activity  

✅ **Weather Data Coupling**  
- Replace simple January/April rules with **real historical weather data**  
- Improve seasonal realism across all variables  

✅ **Advanced Sensor Modeling**  
- Introduce **nonlinear drift** instead of linear trends  
- Add **long-memory noise processes** to capture gradual, environment-driven degradation  






