import pandas as pd
from typing import Dict, Any
from pathlib import Path
from config.settings import config
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataLoader:
    def __init__(self, config=config) -> None:
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV with validation."""
        path = Path(self.config.RAW_DATA_PATH)

        logger.info(f"Loading data from {path}")
        if not path.exists():
            error_msg = f"Data file not found: {path}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            error_msg = f"Failed to read CSV file: {e}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        if df.empty:
            error_msg = "The loaded dataframe is empty."
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        required_columns = [self.config.TARGET_COLUMN, self.config.TEXT_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        # Check for nulls in required columns
        null_counts = df[required_columns].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                logger.warning(f"Found {count} null values in {col}")

        self.validate_dataframe(df)

        logger.info(f"Successfully loaded {len(df)} rows.")
        return df

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate dataframe structure and content."""
        if df.shape[0] == 0:
            raise DataValidationError("Dataframe has 0 rows.")

        if df[self.config.TARGET_COLUMN].isnull().all():
            raise DataValidationError(f"Target column '{self.config.TARGET_COLUMN}' is all nulls.")

        if df[self.config.TEXT_COLUMN].isnull().all():
            raise DataValidationError(f"Text column '{self.config.TEXT_COLUMN}' is all nulls.")

        n_classes = df[self.config.TARGET_COLUMN].nunique()
        if n_classes < 2:
            raise DataValidationError(
                f"Target column must have at least 2 classes. Found {n_classes}."
            )

    def get_target_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Return class distribution."""
        return dict(df[self.config.TARGET_COLUMN].value_counts().to_dict())  # type: ignore

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return comprehensive data summary."""
        return {
            "shape": df.shape,
            "target_distribution": self.get_target_distribution(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }
