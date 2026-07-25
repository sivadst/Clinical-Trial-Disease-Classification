import joblib
import numpy as np
from typing import Dict, Any
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.models.base_model import BaseClassifier
from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SklearnClassifierWrapper(BaseClassifier):
    def __init__(self, model: Any, name: str) -> None:
        self.model = model
        self.name = name

    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)

    def predict(self, X: Any) -> np.ndarray:
        return np.array(self.model.predict(X))  # type: ignore

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.array(self.model.predict_proba(X))  # type: ignore
        logger.warning(
            f"{self.name} does not support predict_proba natively. Falling back to decision_function or mock probabilities."
        )
        # Fallback to decision_function if possible
        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            # convert to pseudo-probabilities
            if decision.ndim == 1:
                prob = 1 / (1 + np.exp(-decision))
                return np.vstack([1 - prob, prob]).T
            else:
                prob = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
                return np.array(prob)  # type: ignore

        raise NotImplementedError(f"{self.name} does not support probability predictions.")

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision_macro": float(
                precision_score(y_test, preds, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

    @property
    def model_name(self) -> str:
        return self.name


class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs) -> BaseClassifier:
        random_state = config.RANDOM_STATE

        if model_name == "MultinomialNB":
            model = MultinomialNB(**kwargs)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(
                class_weight="balanced", random_state=random_state, max_iter=1000, **kwargs
            )
        elif model_name == "RandomForest":
            model = RandomForestClassifier(
                class_weight="balanced", random_state=random_state, n_jobs=-1, **kwargs
            )
        elif model_name == "LinearSVC":
            # LinearSVC needs CalibratedClassifierCV for probability outputs
            base_svc = LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
                dual=False,
                **kwargs,
            )
            model = CalibratedClassifierCV(base_svc, cv=3)
        elif model_name == "XGBoost":
            # XGBoost doesn't use "balanced" natively but we can tune it later or just leave it out for base init.
            # Assuming labels are 0-indexed integers by the time they reach here.
            model = XGBClassifier(
                random_state=random_state, n_jobs=-1, eval_metric="mlogloss", **kwargs
            )
        elif model_name == "VotingClassifier":
            estimators = kwargs.get("estimators")
            if not estimators:
                raise ValueError("VotingClassifier requires 'estimators' argument.")
            model = VotingClassifier(estimators=estimators, voting="soft")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return SklearnClassifierWrapper(model, model_name)
