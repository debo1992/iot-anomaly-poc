import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import torch
import pandas as pd
from datasets.anomaly_dataset import load_dataset
from tests.load_eval_model import load_model_from_mlflow

try:
    import onnxruntime as ort
    onnx_available = True
except ModuleNotFoundError:
    print("ONNX Runtime not installed. Skipping ONNX benchmarks.")
    onnx_available = False

NUM_RUNS = 50  # number of inference runs for timing

# --- Load dataset ---
train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])

config, float_model = load_model_from_mlflow()  # original float model
float_model.eval()

# Load datasets and loader
train_dataset, val_dataset, _, val_loader = load_dataset(config, train_df, val_df)

# --- Quantized model (JIT) ---
quant_model_path = "outputs/models/my_quantized_model_scripted.pt"
quant_model = torch.jit.load(quant_model_path)
quant_model.eval()

# --- Helper function to measure inference time ---
def measure_inference_time(model, loader, device="cpu", num_runs=NUM_RUNS):
    model.to(device)
    times = []
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            X = X.to(device)
            if i >= num_runs:
                break
            start = time.time()
            _ = model(X)
            end = time.time()
            times.append(end - start)
    return sum(times)/len(times)

def measure_onnx_inference_time(onnx_path, loader, num_runs=NUM_RUNS):
    if not onnx_available:
        return None
    ort_session = ort.InferenceSession(onnx_path)
    times = []
    for i, (X, _) in enumerate(loader):
        if i >= num_runs:
            break
        x_np = X[0:1].numpy().astype("float32")
        start = time.time()
        _ = ort_session.run(None, {"input": x_np})
        end = time.time()
        times.append(end - start)
    return sum(times)/len(times)

# --- File sizes ---
onnx_model_path = "outputs/models/my_quantized_model.onnx"
float_model_path = "mlruns/403589896195770437/724a77bb90034332bdb123578c23d6da/artifacts/CNN_run_1758931464_model.pt"  # replace if needed
float_size = os.path.getsize(float_model_path) / 1024**2
quant_size = os.path.getsize(quant_model_path) / 1024**2
onnx_size = os.path.getsize(onnx_model_path) / 1024**2 if onnx_available else None

print(f"Float model size: {float_size:.2f} MB")
print(f"Quantized model size: {quant_size:.2f} MB")
if onnx_available:
    print(f"ONNX model size: {onnx_size:.2f} MB")

# --- Measure inference time ---
float_time = measure_inference_time(float_model, val_loader)
quant_time = measure_inference_time(quant_model, val_loader)

print(f"Average inference time per batch (float): {float_time*1000:.2f} ms")
print(f"Average inference time per batch (quantized): {quant_time*1000:.2f} ms")
if onnx_available:
    onnx_time = measure_onnx_inference_time(onnx_model_path, val_loader)
    print(f"Average inference time per batch (ONNX): {onnx_time*1000:.2f} ms")
