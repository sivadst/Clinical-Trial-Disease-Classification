import re
import joblib
import pandas as pd
import scipy.sparse
from typing import List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from config.settings import config
from src.utils.logger import get_logger
from src.utils.exceptions import PreprocessingError

logger = get_logger(__name__)

# Try downloading NLTK data required
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    def __init__(self, config=config) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.MAX_FEATURES,
            ngram_range=config.NGRAM_RANGE,
            min_df=config.MIN_DF,
            max_df=config.MAX_DF,
        )
        self.lemmatizer = WordNetLemmatizer()

        # English stopwords but keep medical negations
        all_stopwords = set(stopwords.words("english"))
        negations = {"no", "not", "without", "against", "denies", "absent", "nor"}
        self.stop_words = all_stopwords - negations

        self.abbreviations = {
            r"\bpt\b": "patient",
            r"\bhx\b": "history",
            r"\bw/\b": "with",
            r"\bdx\b": "diagnosis",
            r"\btx\b": "treatment",
        }

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        if not isinstance(text, str) or not text.strip():
            return ""

        try:
            text = self.clean_text(text)
            text = self.handle_medical_abbreviations(text)
            tokens = text.split()
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize(tokens)
            return " ".join(tokens)
        except Exception as e:
            logger.error(f"Error during preprocessing text: {e}")
            raise PreprocessingError(f"Failed to preprocess text: {e}")

    def clean_text(self, text: str) -> str:
        """Lowercase, remove special chars, normalize whitespace."""
        text = text.lower()
        # Keep alphanumeric, spaces, and specific punctuation for negations if needed
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def handle_medical_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        for pattern, replacement in self.abbreviations.items():
            text = re.sub(pattern, replacement, text)
        return text

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords but preserve medical negations."""
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize using NLTK WordNet."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def fit_transform(self, texts: pd.Series) -> scipy.sparse.csr_matrix:
        """Fit TF-IDF and return transformed matrix."""
        logger.info("Fitting and transforming TF-IDF vectorizer...")
        processed_texts = texts.apply(self.preprocess)
        return self.vectorizer.fit_transform(processed_texts)

    def transform(self, texts: pd.Series) -> scipy.sparse.csr_matrix:
        """Transform new texts using fitted TF-IDF."""
        logger.info("Transforming texts using fitted TF-IDF vectorizer...")
        processed_texts = texts.apply(self.preprocess)
        return self.vectorizer.transform(processed_texts)

    def save_vectorizer(self, path: str) -> None:
        """Save fitted TF-IDF vectorizer."""
        logger.info(f"Saving vectorizer to {path}")
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path: str) -> None:
        """Load TF-IDF vectorizer."""
        logger.info(f"Loading vectorizer from {path}")
        self.vectorizer = joblib.load(path)

    def save_label_encoder(self, encoder: LabelEncoder, path: str) -> None:
        """Save label encoder."""
        logger.info(f"Saving label encoder to {path}")
        joblib.dump(encoder, path)

    def load_label_encoder(self, path: str) -> Any:
        """Load label encoder."""
        logger.info(f"Loading label encoder from {path}")
        return joblib.load(path)
