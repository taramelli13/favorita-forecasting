"""Centralized project configuration using Pydantic."""

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathConfig(BaseModel):
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_submissions: Path = PROJECT_ROOT / "data" / "submissions"
    models: Path = PROJECT_ROOT / "models"
    configs: Path = PROJECT_ROOT / "configs"
    params_file: Path = PROJECT_ROOT / "params.yaml"


class FeatureConfig(BaseModel):
    lags: list[int] = [16, 17, 18, 19, 20, 21, 28, 30, 60, 90, 180, 365]
    rolling_windows: list[int] = [7, 14, 28, 60, 90, 180, 365]
    rolling_shift: int = 16
    rolling_stats: list[str] = ["mean", "std", "min", "max"]
    use_target_encoding: bool = True
    use_cyclical_encoding: bool = True


class LightGBMConfig(BaseModel):
    objective: str = "tweedie"
    tweedie_variance_power: float = 1.5
    learning_rate: float = 0.02
    num_leaves: int = 255
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_estimators: int = 3000
    early_stopping_rounds: int = 100
    n_jobs: int = -1
    device: str = "gpu"


class XGBoostConfig(BaseModel):
    objective: str = "reg:tweedie"
    tweedie_variance_power: float = 1.5
    learning_rate: float = 0.02
    max_depth: int = 8
    min_child_weight: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_estimators: int = 3000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    device: str = "cuda"


class TrainConfig(BaseModel):
    model_type: str = "lightgbm"
    cv_splits: int = 4
    forecast_horizon: int = 16
    seed: int = 42
    lightgbm: LightGBMConfig = LightGBMConfig()
    xgboost: XGBoostConfig = XGBoostConfig()


class DataConfig(BaseModel):
    train_start_date: str = "2015-01-01"
    train_end_date: str = "2017-08-15"
    test_start_date: str = "2017-08-16"
    test_end_date: str = "2017-08-31"
    min_train_date: str = "2016-08-01"


class TuningConfig(BaseModel):
    n_trials: int = 100
    timeout: int | None = None


class EnsembleConfig(BaseModel):
    method: str = "weighted_average"


class Settings(BaseSettings):
    paths: PathConfig = PathConfig()
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    train: TrainConfig = TrainConfig()
    tuning: TuningConfig = TuningConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    mlflow_tracking_uri: str = "mlruns"

    model_config = {"env_prefix": "FAVORITA_"}


def load_settings() -> Settings:
    """Load settings from params.yaml, falling back to defaults."""
    params_path = PROJECT_ROOT / "params.yaml"
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)
        return Settings(**params)
    return Settings()
