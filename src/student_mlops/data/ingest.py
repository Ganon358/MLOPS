from pathlib import Path
import pandas as pd


def load_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data("data/raw/bangladesh_student_performance.csv")

    print("Dimensions :", df.shape)
    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nAperçu :")
    print(df.head())

    print("\nInfos :")
    print(df.info())

    print("\nValeurs manquantes :")
    print(df.isna().sum())