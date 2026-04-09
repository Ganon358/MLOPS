from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from student_mlops.data.preprocess import (
    load_processed_data,
    get_feature_target_split,
    build_preprocessor,
)


def train_model(train_path: str = "data/processed/train.csv") -> None:
    df = load_processed_data(train_path)
    x_train, y_train = get_feature_target_split(df)

    preprocessor = build_preprocessor(x_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=42))
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student-performance-regression")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(x_train)

        mae = mean_absolute_error(y_train, predictions)
        mse = mean_squared_error(y_train, predictions)
        r2 = r2_score(y_train, predictions)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_r2", r2)

        mlflow.sklearn.log_model(best_model, "model")

        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, "models/model.joblib")

        print("Entraînement terminé.")
        print("Meilleurs paramètres :", grid_search.best_params_)
        print("MAE :", mae)
        print("MSE :", mse)
        print("R2 :", r2)


if __name__ == "__main__":
    train_model()