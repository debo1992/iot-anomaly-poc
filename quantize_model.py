import torch
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.fx import symbolic_trace
import pandas as pd

from tests.load_eval_model import load_model_from_mlflow, prediction_model, log_confusion_matrix, log_pr_curves
from datasets.anomaly_dataset import load_dataset, get_calibration_loader


def quantize_model(model, config, val_loader=None, device="cpu"):
    """
    Quantize model based on type in config.
    """
    model_type = config.get("model_type", "").upper()
    model.to(device)
    model.eval()

    if model_type in ["LSTM", "TRANSFORMER"]:
        # ✅ Dynamic quantization (linear layers only)
        print("Applying Dynamic Quantization for:", model_type)
        return tq.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    elif model_type in ["CNN", "CNN_DILATION", "TCN"]:
        if val_loader is None:
            raise ValueError("val_loader is required for static quantization of CNN/TCN")

        print("Applying Post-Training Static Quantization for:", model_type)

        # Symbolic trace the model for FX workflow
        traced_model = symbolic_trace(model)

        # Attach qconfig
        qconfig = tq.get_default_qconfig("fbgemm")
        example_input = torch.randn(1, config["window_size"], train_dataset.X.shape[2]).to(device)
        prepared_model = prepare_fx(traced_model, {"": qconfig}, example_inputs=(example_input,))

        # Calibration pass
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(device)
                prepared_model(X)

        # Convert to quantized model
        quantized_model = convert_fx(prepared_model)
        return quantized_model

    else:
        print(f"Unknown model_type={model_type}, defaulting to dynamic quantization")
        return tq.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )


def save_quantized_model(quantized_model, config, example_input, path="outputs/models/quantized.pt"):
    """
    Save quantized model in TorchScript and ONNX formats.
    """
    quantized_model.eval()
    torch.save(quantized_model.state_dict(), "mlruns/403589896195770437/724a77bb90034332bdb123578c23d6da/artifacts/quantized_state_dict.pt")
    # TorchScript (script is safer than trace for control flow)
    scripted = torch.jit.script(quantized_model)
    torch.jit.save(scripted, path.replace(".pt", "_scripted.pt"))

    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])
    config, model, model_name = load_model_from_mlflow()
    _, _, _, val_loader = load_dataset(config, train_df, val_df)
    labels, predicted_class, predicted_probabilities = prediction_model(model, val_loader)
    log_confusion_matrix(labels, predicted_class,
                     class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], normalized=True,
                     artifact_name="cf_matrix_quantized.png")
    log_pr_curves(labels, predicted_probabilities, class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"], artifact_name="pr_quantized.png")

    # ONNX
    torch.onnx.export(
        model, example_input.cpu(), path.replace(".pt", ".onnx"),
        export_params=True, opset_version=17,
        do_constant_folding=True, input_names=["input"], output_names=["output"]
    )
    print("Quantized model saved at:", path)


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("datasets/data/train_all.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("datasets/data/val_all.csv", parse_dates=["timestamp"])

    # Load best model + config from MLflow
    config, model, model_name = load_model_from_mlflow()
    model.eval()

    # Datasets + loaders
    train_dataset, val_dataset, _, val_loader = load_dataset(config, train_df, val_df)

    # Baseline evaluation before quantization
    labels, predicted_class, predicted_probabilities = prediction_model(model, val_loader)
    log_confusion_matrix(labels, predicted_class,
                         class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"],
                         normalized=True, artifact_name="cf_matrix_"+model_name+".png")
    log_pr_curves(labels, predicted_probabilities,
                  class_names=["Normal", "Temp", "Humid", "Fridge", "Door", "Fire"],
                  artifact_name="pr_"+model_name+".png")

    # Quantization
    calib_loader = get_calibration_loader(train_dataset, frac=0.1, batch_size=64)
    quantized_model = quantize_model(model, config, val_loader=calib_loader, device="cpu")

    # Save quantized versions
    example_input = torch.randn(1, config["window_size"], train_dataset.X.shape[2])
    save_quantized_model(quantized_model, config, example_input, path="outputs/models/"+model_name+"_quantized.pt")
