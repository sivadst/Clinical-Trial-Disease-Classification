import pytest
import pandas as pd
from src.data.preprocessor import TextPreprocessor
import scipy.sparse


@pytest.fixture
def preprocessor():
    class MockConfig:
        MAX_FEATURES = 100
        NGRAM_RANGE = (1, 1)
        MIN_DF = 1
        MAX_DF = 1.0

    return TextPreprocessor(config=MockConfig())


def test_preprocess_empty_string(preprocessor):
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess("   ") == ""
    assert preprocessor.preprocess(None) == ""


def test_preprocess_preserves_negation(preprocessor):
    text = "Patient denies any history of cancer and has no evidence of disease."
    processed = preprocessor.preprocess(text)
    assert "denies" in processed
    assert "no" in processed


def test_preprocess_medical_abbreviations(preprocessor):
    text = "The pt hx shows a dx of type 2 diabetes with no tx."
    processed = preprocessor.preprocess(text)
    assert "patient" in processed
    assert "history" in processed
    assert "diagnosis" in processed
    assert "treatment" in processed


def test_tf_idf_consistent_vocabulary(preprocessor):
    train_texts = pd.Series(["first clinical trial", "second clinical trial for disease"])

    test_texts = pd.Series(["third clinical trial"])

    X_train = preprocessor.fit_transform(train_texts)
    vocab_len = len(preprocessor.vectorizer.vocabulary_)

    X_test = preprocessor.transform(test_texts)

    assert X_train.shape[1] == vocab_len
    assert X_test.shape[1] == vocab_len
    assert isinstance(X_train, scipy.sparse.csr_matrix)
    assert isinstance(X_test, scipy.sparse.csr_matrix)
