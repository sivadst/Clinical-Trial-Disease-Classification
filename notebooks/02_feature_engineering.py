import sys
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Feature Engineering Deep Dive"),
    nbf.v4.new_code_cell("""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import config
from src.data.preprocessor import TextPreprocessor

df = pd.read_csv(config.RAW_DATA_PATH).head(5000) # subset for speed
"""),
    nbf.v4.new_markdown_cell("## Initialize Preprocessor and TF-IDF extraction"),
    nbf.v4.new_code_cell("""
preprocessor = TextPreprocessor()
X = preprocessor.fit_transform(df[config.TEXT_COLUMN])
print(f"Original Data Shape: {df.shape}")
print(f"TF-IDF Matrix Shape: {X.shape}")
"""),
    nbf.v4.new_markdown_cell("## Analyze Vocabulary and Important Features"),
    nbf.v4.new_code_cell("""
feature_names = preprocessor.vectorizer.get_feature_names_out()
tfidf_sums = X.sum(axis=0).A1
top_indices = tfidf_sums.argsort()[::-1][:20]
top_features = [feature_names[i] for i in top_indices]
top_scores = [tfidf_sums[i] for i in top_indices]

plt.figure(figsize=(10,6))
sns.barplot(x=top_scores, y=top_features, palette='viridis')
plt.title('Top 20 TF-IDF Features Overall')
plt.xlabel('TF-IDF Sum')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
"""),
]

with open('notebooks/02_feature_engineering.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 02 generated.")
