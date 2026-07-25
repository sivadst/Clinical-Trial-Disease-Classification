import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.eda_plots import (
    plot_class_distribution,
    plot_text_length_distribution,
    plot_missing_values,
    plot_wordcloud,
    plot_top_words
)
from config.settings import config
import nbformat as nbf

# Define paths
data_path = config.RAW_DATA_PATH
reports_dir = config.REPORTS_DIR
figures_dir = config.FIGURES_DIR

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(data_path)

# Notebook generation logic
nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Exploratory Data Analysis"),
    nbf.v4.new_code_cell("""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.eda_plots import plot_class_distribution, plot_text_length_distribution, plot_missing_values, plot_wordcloud, plot_top_words
from config.settings import config

df = pd.read_csv(config.RAW_DATA_PATH)
"""),
    nbf.v4.new_markdown_cell("## Section A: Dataset Overview"),
    nbf.v4.new_code_cell("""
print(f"Shape: {df.shape}")
print(df.dtypes)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
plot_missing_values(df)
plt.show()
"""),
    nbf.v4.new_markdown_cell("## Section B: Target Variable Analysis"),
    nbf.v4.new_code_cell("""
class_counts = df[config.TARGET_COLUMN].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
print(f"Target variable: {config.TARGET_COLUMN} with {df[config.TARGET_COLUMN].nunique()} classes. Imbalance ratio: {imbalance_ratio:.2f}")

plot_class_distribution(df, target_col=config.TARGET_COLUMN)
plt.show()
"""),
    nbf.v4.new_markdown_cell("## Section C: Text Feature Analysis"),
    nbf.v4.new_code_cell("""
plot_text_length_distribution(df, text_col=config.TEXT_COLUMN)
plt.show()

plot_top_words(df, text_col=config.TEXT_COLUMN, n_top=20)
plt.show()
"""),
    nbf.v4.new_markdown_cell("## Section D & E: Word clouds & Quality Report"),
    nbf.v4.new_code_cell("""
top_classes = df[config.TARGET_COLUMN].value_counts().head(3).index.tolist()
for c in top_classes:
    plot_wordcloud(df, text_col=config.TEXT_COLUMN, target_col=config.TARGET_COLUMN, class_name=c)
    plt.show()
""")
]

with open('notebooks/01_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated.")
