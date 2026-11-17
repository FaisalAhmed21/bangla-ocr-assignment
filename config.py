import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    dataset_path: str = "BanglaLekha-Isolated/Images"
    labels_json_path: str = "labels.json"


class TrainSettings(BaseSettings):
    image_size: int = 64
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    sample_per_class: int | None = None
    model_save_path: str = "models/model.keras"
    save_json_path: str = "models/labels.json"
    save_summary_path: str = "models/train_history.csv"


class PredictSettings(BaseSettings):
    model_path: str = "models/model.keras"
    labels_json_path: str = "labels.json"
    image_size: int = 32  # Changed from 64 to 32 to match your model.keras


class MLflowSettings(BaseSettings):
    tracking_uri: str = "sqlite:///artifacts/mlflow/mlflow.db"
    registry_uri: str = "sqlite:///artifacts/mlflow/mlflow.db"
    experiment_name: str = "bangla-ocr"


class Settings(BaseSettings):
    seed: int = 42
    data: DataSettings = DataSettings()
    train: TrainSettings = TrainSettings()
    predict: PredictSettings = PredictSettings()
    mlflow: MLflowSettings = MLflowSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )


settings = Settings()