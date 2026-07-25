import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    RAW_DATA_PATH: str = str(BASE_DIR / "data" / "clinical_trials_raw_patient2trial_conditions.csv")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "data" / "processed" / "processed_data.csv")
    MODELS_DIR: str = str(BASE_DIR / "models")
    REPORTS_DIR: str = str(BASE_DIR / "reports")
    FIGURES_DIR: str = str(BASE_DIR / "reports" / "figures")

    # Target and Features
    TARGET_COLUMN: str = "source_condition_query"
    TEXT_COLUMN: str = "brief_summary"

    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.125 # 0.125 of train yields 70/10/20 overall split
    N_FOLDS: int = 5

    # TF-IDF parameters
    MAX_FEATURES: int = 10000
    NGRAM_RANGE: tuple = (1, 2)
    MIN_DF: int = 2
    MAX_DF: float = 0.95

    # Classifiers to train
    CLASSIFIERS: list = [
        "MultinomialNB",
        "LogisticRegression",
        "RandomForest",
        "LinearSVC",
        "XGBoost"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate settings
config = Settings()
