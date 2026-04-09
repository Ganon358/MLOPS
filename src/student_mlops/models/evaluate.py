from pathlib import Path
import joblib
import mlflow
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from student_mlops.data.preprocess import load_processed_data, get_feature_target_split


def evaluate_model(model_path: str, data_path: str, batch_name: str) -> dict:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_file}")

    model = joblib.load(model_file)

    df = load_processed_data(data_path)
    x_test, y_test = get_feature_target_split(df)

    predictions = model.predict(x_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student-performance-regression")

    with mlflow.start_run(run_name=f"evaluation_{batch_name}"):
        mlflow.log_param("evaluated_batch", batch_name)
        mlflow.log_param("data_path", data_path)
        mlflow.log_metric(f"{batch_name}_mae", metrics["mae"])
        mlflow.log_metric(f"{batch_name}_mse", metrics["mse"])
        mlflow.log_metric(f"{batch_name}_r2", metrics["r2"])

    print(f"\nÉvaluation sur {batch_name}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"MSE : {metrics['mse']:.4f}")
    print(f"R2  : {metrics['r2']:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate_model("models/model.joblib", "data/processed/batch_1.csv", "batch_1")
    evaluate_model("models/model.joblib", "data/processed/batch_2.csv", "batch_2")