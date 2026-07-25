import os
import sys
from pathlib import Path
import time
from sklearn.preprocessing import LabelEncoder

# Insert path before custom module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import config
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.data.splitter import DataSplitter
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    start_time = time.time()
    logger.info("Starting training pipeline...")

    # Ensure directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # 1. Load Data
    loader = DataLoader()
    df = loader.load_raw_data()

    # 2. Split Data
    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split(df)

    # 3. Preprocess Text
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(train_df[config.TEXT_COLUMN])
    X_val = preprocessor.transform(val_df[config.TEXT_COLUMN])
    X_test = preprocessor.transform(test_df[config.TEXT_COLUMN])

    preprocessor.save_vectorizer(os.path.join(config.MODELS_DIR, "tfidf_vectorizer.pkl"))

    # 4. Encode Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[config.TARGET_COLUMN])
    y_val = label_encoder.transform(val_df[config.TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[config.TARGET_COLUMN])

    preprocessor.save_label_encoder(
        label_encoder, os.path.join(config.MODELS_DIR, "label_encoder.pkl")
    )

    # 5. Train & Evaluate Models
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    results = {}
    best_f1 = -1
    best_model = None
    best_model_name = ""

    # Train all base models
    trained_models = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Validate base models using cross-validation (or validation set) to find top 2
    val_scores = {}
    for name, model in trained_models.items():
        cv_res = trainer.cross_validate(model, X_train, y_train)
        val_scores[name] = cv_res["f1_mean"]
        logger.info(f"{name} CV F1 Mean: {cv_res['f1_mean']:.4f}")

    top_2_names = sorted(val_scores, key=val_scores.get, reverse=True)[:2]
    logger.info(f"Top 2 models for tuning: {top_2_names}")

    # Tune top 2
    tuned_models = {}
    for name in top_2_names:
        tuned_model = trainer.hyperparameter_search(trained_models[name], X_train, y_train)
        tuned_model.train(X_train, y_train)
        tuned_models[f"{name}_tuned"] = tuned_model

    all_final_models = {**trained_models, **tuned_models}

    # Evaluate all models on test set
    for name, model in all_final_models.items():
        eval_metrics = evaluator.evaluate_model(model, X_test, y_test, label_encoder)
        results[name] = eval_metrics

        f1 = eval_metrics["f1_macro"]
        logger.info(f"Test F1 Macro for {name}: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    logger.info(f"Best model: {best_model_name} with F1 Macro: {best_f1:.4f}")

    # Save best model
    best_model.save(os.path.join(config.MODELS_DIR, "best_model.pkl"))

    # 6. Generate Report and Plots
    evaluator.plot_confusion_matrix(
        results[best_model_name]["confusion_matrix"],
        label_encoder.classes_.tolist(),
        os.path.join(config.FIGURES_DIR, "confusion_matrix.png"),
    )

    evaluator.plot_model_comparison(
        results, "f1_macro", os.path.join(config.FIGURES_DIR, "model_comparison_f1.png")
    )

    # Write report
    report_md = "# Model Training Report\n\n"
    report_md += f"## Best Model: {best_model_name}\n"
    report_md += f"- **Test F1 Macro:** {best_f1:.4f}\n"
    report_md += f"- **Test Accuracy:** {results[best_model_name]['accuracy']:.4f}\n\n"

    report_md += "## Model Comparison\n"
    report_md += "| Model | Accuracy | F1 Macro |\n"
    report_md += "|---|---|---|\n"
    for name, res in results.items():
        report_md += f"| {name} | {res['accuracy']:.4f} | {res['f1_macro']:.4f} |\n"

    report_md += "\n![F1 Comparison](figures/model_comparison_f1.png)\n"
    report_md += "\n## Best Model Confusion Matrix\n"
    report_md += "![Confusion Matrix](figures/confusion_matrix.png)\n"

    with open(os.path.join(config.REPORTS_DIR, "model_report.md"), "w") as f:
        f.write(report_md)

    # Generate requested artifacts
    import json

    with open(os.path.join(config.MODELS_DIR, "metrics.json"), "w") as f:
        # Convert dict to handle numpy floats
        metrics_clean = {
            k: {
                inner_k: (float(inner_v) if isinstance(inner_v, (float, int)) else inner_v)
                for inner_k, inner_v in v.items()
                if inner_k != "classification_report"
            }
            for k, v in results.items()
        }
        json.dump(metrics_clean, f, indent=4)

    with open(os.path.join(config.MODELS_DIR, "classification_report.txt"), "w") as f:
        from sklearn.metrics import classification_report

        y_test_str = label_encoder.inverse_transform(y_test)
        preds_str = label_encoder.inverse_transform(best_model.predict(X_test))
        f.write(classification_report(y_test_str, preds_str, zero_division=0))

    duration = time.time() - start_time
    logger.info(f"Training pipeline completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
