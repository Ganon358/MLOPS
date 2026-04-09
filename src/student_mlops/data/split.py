from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

from student_mlops.data.ingest import load_data


def split_and_save_data(
    input_path: str,
    output_dir: str,
    random_state: int = 42
) -> None:
    df = load_data(input_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        shuffle=True
    )

    batch_1_df, batch_2_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        shuffle=True
    )

    train_df.to_csv(output_path / "train.csv", index=False)
    batch_1_df.to_csv(output_path / "batch_1.csv", index=False)
    batch_2_df.to_csv(output_path / "batch_2.csv", index=False)

    print("Découpage terminé.")
    print(f"Train : {train_df.shape}")
    print(f"Batch 1 : {batch_1_df.shape}")
    print(f"Batch 2 : {batch_2_df.shape}")


if __name__ == "__main__":
    split_and_save_data(
        input_path="data/raw/bangladesh_student_performance.csv",
        output_dir="data/processed"
    )