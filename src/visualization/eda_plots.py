import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from pathlib import Path
from wordcloud import WordCloud


def plot_class_distribution(
    df: pd.DataFrame, target_col: str, save_path: Optional[str] = None, figsize: tuple = (10, 6)
) -> plt.Figure:
    """Plot class distribution as a bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    counts = df[target_col].value_counts()

    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title(f"Class Distribution of {target_col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("Class")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_text_length_distribution(
    df: pd.DataFrame, text_col: str, save_path: Optional[str] = None, figsize: tuple = (10, 6)
) -> plt.Figure:
    """Plot the distribution of text lengths."""
    fig, ax = plt.subplots(figsize=figsize)
    lengths = df[text_col].dropna().astype(str).str.len()

    sns.histplot(lengths, bins=50, kde=True, ax=ax, color="skyblue")
    ax.set_title(f"Text Length Distribution for {text_col}")
    ax.set_xlabel("Length (characters)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_missing_values(
    df: pd.DataFrame, save_path: Optional[str] = None, figsize: tuple = (12, 8)
) -> plt.Figure:
    """Plot missing values as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_wordcloud(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    class_name: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """Plot word cloud for a specific class."""
    fig, ax = plt.subplots(figsize=figsize)
    text_data = " ".join(df[df[target_col] == class_name][text_col].dropna().astype(str).tolist())

    wordcloud = WordCloud(
        width=800, height=800, background_color="white", min_font_size=10
    ).generate(text_data)

    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud for {class_name}")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_top_words(
    df: pd.DataFrame,
    text_col: str,
    n_top: int = 20,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot top N words in the text column."""
    from collections import Counter
    import re

    fig, ax = plt.subplots(figsize=figsize)
    text_data = " ".join(df[text_col].dropna().astype(str).tolist()).lower()
    words = re.findall(r"\b[a-z]{3,}\b", text_data)
    counter = Counter(words)

    common_words = dict(counter.most_common(n_top))

    sns.barplot(x=list(common_words.values()), y=list(common_words.keys()), ax=ax, palette="mako")
    ax.set_title(f"Top {n_top} Words in {text_col}")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig
