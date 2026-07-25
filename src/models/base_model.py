from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseClassifier(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, X_val: Any = None, y_val: Any = None) -> None:
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
