"""Stage 2: Train, evaluate and register the v1 model using MLflow."""

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_TRAIN = Path("data/raw/train.csv")
CLEAN_TRAIN = Path("data/processed/train_clean.csv")
LATER_FILE = Path("data/raw/later.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "model.joblib"

TARGET = "gb_per_day"

NUMERIC_FEATURES = [
    "age",
    "screen_hours",
    "apps_installed",
    "plan_gb_limit",
    "video_minutes",
    "wifi_hours",
]

CATEGORICAL_FEATURES = ["phone_tier", "circle"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPERIMENT_NAME = "G203_MP269_Stage2"
REGISTERED_MODEL_NAME = "G203_MP269_v1"


def calculate_sha256(file_path):
    """Calculate SHA-256 of the original training CSV."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def build_pipeline(model):
    """Create preprocessing and regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model),
        ]
    )


def main():
    print("=" * 65)
    print("STAGE 2 - V1 MODEL TRAINING AND MLFLOW TRACKING")
    print("=" * 65)

    # Validate input files.
    for file_path in [RAW_TRAIN, CLEAN_TRAIN, LATER_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load cleaned training data and clean later evaluation data.
    train_df = pd.read_csv(CLEAN_TRAIN)
    later_df = pd.read_csv(LATER_FILE)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_later = later_df[FEATURES]
    y_later = later_df[TARGET]

    print(f"\nTraining dataset: {X_train.shape}")
    print(f"Later dataset: {X_later.shape}")

    # Hash the ORIGINAL raw training file, not the cleaned file.
    data_sha256 = calculate_sha256(RAW_TRAIN)

    print(f"\nOriginal train.csv SHA-256: {data_sha256}")

    # Configure MLflow with SQLite tracking.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Three different model configurations.
    experiments = [
        {
            "name": "random_forest_200",
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        {
            "name": "random_forest_300",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
        },
        {
            "name": "random_forest_400",
            "n_estimators": 400,
            "max_depth": 8,
            "min_samples_leaf": 3,
        },
    ]

    results = []

    for config in experiments:
        print("\n" + "-" * 65)
        print(f"Running experiment: {config['name']}")
        print("-" * 65)

        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            random_state=42,
            n_jobs=-1,
        )

        pipeline = build_pipeline(model)

        with mlflow.start_run(run_name=config["name"]) as run:
            # Log parameters for every run.
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", config["max_depth"])
            mlflow.log_param(
                "min_samples_leaf", config["min_samples_leaf"]
            )
            mlflow.log_param("random_state", 42)
            mlflow.log_param("training_rows", len(train_df))
            mlflow.log_param("evaluation_rows", len(later_df))

            # Required: original raw train.csv hash in EVERY run.
            mlflow.log_param("data_sha256", data_sha256)

            # Train only on cleaned training data.
            pipeline.fit(X_train, y_train)

            # Evaluate on the later dataset.
            predictions = pipeline.predict(X_later)
            mae = float(mean_absolute_error(y_later, predictions))

            # Required MAE metric.
            mlflow.log_metric("mae_later", mae)

            # Save the model artifact in this MLflow run.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "name": config["name"],
                    "run_id": run.info.run_id,
                    "mae": mae,
                    "pipeline": pipeline,
                }
            )

            print(f"Run ID: {run.info.run_id}")
            print(f"MAE on later.csv: {mae:.6f}")

    # Select the best model using lowest later.csv MAE.
    best_result = min(results, key=lambda item: item["mae"])

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPARISON")
    print("=" * 65)

    for result in results:
        print(f"{result['name']}: MAE = {result['mae']:.6f}")

    print("\nBest model:", best_result["name"])
    print(f"Best MAE: {best_result['mae']:.6f}")

    # Register the best run's model.
    model_uri = f"runs:/{best_result['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    # Save a local copy for the project.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_result["pipeline"], MODEL_FILE)

    print("\n" + "=" * 65)
    print("V1 MODEL REGISTRATION COMPLETE")
    print("=" * 65)
    print("Experiment:", EXPERIMENT_NAME)
    print("Registered model:", REGISTERED_MODEL_NAME)
    print("Model version:", registered_model.version)
    print("Model alias: champion")
    print("Best run ID:", best_result["run_id"])
    print("Local model:", MODEL_FILE)
    print("MLflow database: mlflow.db")
    print("\nStage 2 v1 training completed successfully.")


if __name__ == "__main__":
    main()
