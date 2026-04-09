from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "HSC_Result"
DROP_COLUMNS = ["Student_ID"]


def load_processed_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    return pd.read_csv(path)


def get_feature_target_split(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET_COLUMN}' est absente du dataset.")

    x = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN]

    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_features = x.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = x.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


if __name__ == "__main__":
    df = load_processed_data("data/processed/train.csv")
    x, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(x)

    print("Cible :", TARGET_COLUMN)
    print("Shape X :", x.shape)
    print("Shape y :", y.shape)
    print("Colonnes catégorielles :", x.select_dtypes(include=["object"]).columns.tolist())
    print("Colonnes numériques :", x.select_dtypes(exclude=["object"]).columns.tolist())

    x_transformed = preprocessor.fit_transform(x)
    print("Prétraitement terminé.")
    print("Shape transformée :", x_transformed.shape)