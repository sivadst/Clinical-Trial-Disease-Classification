import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    def evaluate_model(
        self, model: Any, X_test: Any, y_test: Any, label_encoder: Any
    ) -> Dict[str, Any]:
        """Comprehensive evaluation."""
        logger.info(f"Evaluating model: {model.model_name}")

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
        rec_weighted = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1_mac = f1_score(y_test, preds, average="macro", zero_division=0)
        f1_wei = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Need string labels for report
        y_test_str = label_encoder.inverse_transform(y_test)
        preds_str = label_encoder.inverse_transform(preds)

        report = classification_report(y_test_str, preds_str, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, preds)

        return {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "precision_weighted": float(prec_weighted),
            "recall_macro": float(rec_macro),
            "recall_weighted": float(rec_weighted),
            "f1_macro": float(f1_mac),
            "f1_weighted": float(f1_wei),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
        }

    def plot_confusion_matrix(
        self, conf_matrix: list, classes: list, save_path: str, figsize: tuple = (10, 8)
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metric: str,
        save_path: str,
        figsize: tuple = (10, 6),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)

        models = list(results_dict.keys())
        scores = [results_dict[m][metric] for m in models]

        sns.barplot(x=scores, y=models, ax=ax, palette="viridis")
        ax.set_title(f"Model Comparison: {metric}")
        ax.set_xlabel(metric.capitalize())
        ax.set_xlim(0, 1.0)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
