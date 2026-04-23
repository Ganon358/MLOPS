from pathlib import Path
import pandas as pd

from student_mlops.models.evaluate import evaluate_model
from student_mlops.models.train import train_model


R2_THRESHOLD = 0.50


def merge_datasets(train_path: str, new_data_path: str, output_path: str) -> str:
    train_df = pd.read_csv(train_path)
    new_data_df = pd.read_csv(new_data_path)

    updated_df = pd.concat([train_df, new_data_df], ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(output_path, index=False)

    print(f"Dataset fusionné sauvegardé : {output_path}")
    print(f"Taille du dataset fusionné : {updated_df.shape}")

    return output_path


def retrain_if_needed(
    model_path: str = "models/model.joblib",
    train_path: str = "data/processed/train.csv",
    new_data_path: str = "data/processed/batch_1.csv",
    combined_output_path: str = "data/processed/train_updated.csv",
    batch_name: str = "batch_1",
    r2_threshold: float = R2_THRESHOLD,
) -> bool:
    metrics = evaluate_model(
        model_path=model_path,
        data_path=new_data_path,
        batch_name=batch_name,
    )

    if metrics["r2"] >= r2_threshold:
        print(
            f"\nPas de réentraînement nécessaire : "
            f"R2={metrics['r2']:.4f} >= seuil={r2_threshold:.2f}"
        )
        return False

    print(
        f"\nRéentraînement déclenché : "
        f"R2={metrics['r2']:.4f} < seuil={r2_threshold:.2f}"
    )

    updated_train_path = merge_datasets(
        train_path=train_path,
        new_data_path=new_data_path,
        output_path=combined_output_path,
    )

    train_model(train_path=updated_train_path)

    print("\nRéentraînement terminé. Le modèle a été mis à jour.")
    return True


if __name__ == "__main__":
    retrain_if_needed(
        model_path="models/model.joblib",
        train_path="data/processed/train.csv",
        new_data_path="data/processed/batch_1.csv",
        combined_output_path="data/processed/train_updated.csv",
        batch_name="batch_1",
        r2_threshold=0.50,
    )