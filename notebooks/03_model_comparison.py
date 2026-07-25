import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Model Comparison and Evaluation"),
    nbf.v4.new_code_cell("""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from config.settings import config
from src.data.loader import DataLoader
from src.data.splitter import DataSplitter

loader = DataLoader()
df = loader.load_raw_data()
train, val, test = DataSplitter().split(df)
"""),
    nbf.v4.new_markdown_cell("## Load Artifacts"),
    nbf.v4.new_code_cell("""
vectorizer = joblib.load(os.path.join(config.MODELS_DIR, 'tfidf_vectorizer.pkl'))
encoder = joblib.load(os.path.join(config.MODELS_DIR, 'label_encoder.pkl'))
model = joblib.load(os.path.join(config.MODELS_DIR, 'best_model.pkl'))

from src.data.preprocessor import TextPreprocessor
preprocessor = TextPreprocessor()

# Quick test transform
processed_texts = test[config.TEXT_COLUMN].head(100).apply(preprocessor.preprocess)
X_test_small = vectorizer.transform(processed_texts)
y_test_small = test[config.TARGET_COLUMN].head(100)
"""),
    nbf.v4.new_markdown_cell("## Predict and Evaluate"),
    nbf.v4.new_code_cell("""
preds = model.predict(X_test_small)
preds_labels = encoder.inverse_transform(preds)

print(classification_report(y_test_small, preds_labels))
""")
]

with open('notebooks/03_model_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 03 generated.")
