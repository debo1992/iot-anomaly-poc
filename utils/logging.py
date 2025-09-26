# evaluate_metrics.py
import time
import wandb
import mlflow

def log_intialize(run_name, project_name = "iot-anomaly-detection", config = None):
        wandb.init(project=project_name, config=config)
        config = wandb.config
        
        # Set up MLflow
        mlflow.set_experiment("IoT_Anomaly_Models")
        with mlflow.start_run(run_name = run_name):
            mlflow.log_params(dict(config))
        return run_name

def log_loss_accuracy(epoch, train_loss, train_acc, val_loss, val_acc):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
