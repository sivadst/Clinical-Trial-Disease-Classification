import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint
import pandas as pd

from src.models.base_model import BaseClassifier
from src.models.classifiers import ModelFactory
from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config=config) -> None:
        self.config = config

    def train_all_models(
        self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None
    ) -> Dict[str, BaseClassifier]:
        """Train all configured models."""
        models = {}
        for clf_name in self.config.CLASSIFIERS:
            try:
                model = ModelFactory.get_model(clf_name)
                model.train(X_train, y_train, X_val, y_val)
                models[clf_name] = model
            except Exception as e:
                logger.error(f"Failed to train {clf_name}: {e}")

        return models

    def cross_validate(self, model: BaseClassifier, X: Any, y: Any) -> Dict[str, float]:
        """Perform stratified k-fold cross-validation."""
        logger.info(f"Running cross validation for {model.model_name}")
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE
        )

        metrics: Dict[str, List[float]] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        X_arr = (
            X
            if isinstance(X, (np.ndarray, pd.DataFrame, pd.Series)) or hasattr(X, "toarray")
            else np.array(X)
        )
        y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)

        # Determine if it's a sparse matrix
        is_sparse = hasattr(X_arr, "tocsr")

        for train_idx, val_idx in skf.split(X_arr, y_arr):
            if is_sparse:
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
            else:
                X_tr, X_va = (
                    X_arr.iloc[train_idx] if hasattr(X_arr, "iloc") else X_arr[train_idx]
                ), (X_arr.iloc[val_idx] if hasattr(X_arr, "iloc") else X_arr[val_idx])

            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

            model_clone = ModelFactory.get_model(model.model_name)
            model_clone.train(X_tr, y_tr)
            preds = model_clone.predict(X_va)

            metrics["accuracy"].append(float(accuracy_score(y_va, preds)))
            metrics["precision"].append(
                float(precision_score(y_va, preds, average="macro", zero_division=0))
            )
            metrics["recall"].append(
                float(recall_score(y_va, preds, average="macro", zero_division=0))
            )
            metrics["f1"].append(float(f1_score(y_va, preds, average="macro", zero_division=0)))

        return {
            "accuracy_mean": float(np.mean(metrics["accuracy"])),
            "accuracy_std": float(np.std(metrics["accuracy"])),
            "precision_mean": float(np.mean(metrics["precision"])),
            "precision_std": float(np.std(metrics["precision"])),
            "recall_mean": float(np.mean(metrics["recall"])),
            "recall_std": float(np.std(metrics["recall"])),
            "f1_mean": float(np.mean(metrics["f1"])),
            "f1_std": float(np.std(metrics["f1"])),
        }

    def hyperparameter_search(
        self, model: BaseClassifier, X_train: Any, y_train: Any
    ) -> BaseClassifier:
        """Run RandomizedSearchCV for a model."""
        logger.info(f"Running hyperparameter search for {model.model_name}")

        param_distributions = {}

        if model.model_name == "LogisticRegression":
            param_distributions = {"C": uniform(loc=0.1, scale=9.9), "penalty": ["l2"]}
        elif model.model_name == "RandomForest":
            param_distributions = {
                "n_estimators": randint(50, 200),
                "max_depth": [None, 10, 20, 30],
            }
        elif model.model_name == "MultinomialNB":
            param_distributions = {"alpha": uniform(loc=0.01, scale=1.0)}
        elif model.model_name == "LinearSVC":
            # Can't easily grid search CalibratedClassifierCV, skip for simplicity unless needed
            logger.info(
                "Skipping hyperparameter search for LinearSVC (wrapped in CalibratedClassifierCV)."
            )
            return model
        elif model.model_name == "XGBoost":
            param_distributions = {
                "learning_rate": uniform(0.01, 0.3),
                "max_depth": randint(3, 10),
                "n_estimators": randint(50, 200),
            }
        else:
            logger.warning(
                f"No param distribution defined for {model.model_name}. Returning original model."
            )
            return model

        if not hasattr(model, "model"):
            raise ValueError(f"Model {model.model_name} doesn't have an underlying sklearn model.")
        sk_model = getattr(model, "model")

        search = RandomizedSearchCV(
            sk_model,
            param_distributions,
            n_iter=5,  # Reduced from 20 to 5 for speed during this exercise
            scoring="f1_macro",
            cv=3,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        logger.info(f"Best parameters for {model.model_name}: {search.best_params_}")

        # Return new wrapper with best model
        # sk_model holds the underlying wrapped model, so we know model has it.
        # It's an instance of SklearnClassifierWrapper or similar.
        return ModelFactory.get_model(model.model_name, **search.best_params_)
