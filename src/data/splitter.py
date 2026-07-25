import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    def __init__(self, config=config) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train/val/test with stratification."""
        logger.info(
            f"Splitting data with TEST_SIZE={self.config.TEST_SIZE} and VAL_SIZE={self.config.VAL_SIZE}"
        )

        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN]

        # Split into train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE, stratify=y
        )

        # Split train_val into train and val
        # To get the desired validation ratio of original data, we adjust the validation size.
        # e.g., if we want 70/10/20, test=0.2, val=0.1 of original -> val_size_of_train_val = 0.1 / 0.8 = 0.125
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.config.VAL_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_train_val,
        )

        # Re-attach target for returning
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        logger.info(
            f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}"
        )

        return train_df, val_df, test_df

    def get_split_statistics(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Return class distribution for each split."""
        stats = {
            "train": train[self.config.TARGET_COLUMN].value_counts(normalize=True).to_dict(),
            "val": val[self.config.TARGET_COLUMN].value_counts(normalize=True).to_dict(),
            "test": test[self.config.TARGET_COLUMN].value_counts(normalize=True).to_dict(),
        }

        # Check stratification (±2%)
        classes = train[self.config.TARGET_COLUMN].unique()
        for c in classes:
            train_prop = stats["train"].get(c, 0)
            val_prop = stats["val"].get(c, 0)
            test_prop = stats["test"].get(c, 0)

            if abs(train_prop - val_prop) > 0.02:
                logger.warning(f"Stratification mismatch for class {c} between train and val > 2%")
            if abs(train_prop - test_prop) > 0.02:
                logger.warning(f"Stratification mismatch for class {c} between train and test > 2%")

        return stats
