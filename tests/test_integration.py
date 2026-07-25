import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.data.preprocessor import TextPreprocessor
from src.models.classifiers import ModelFactory
from src.models.evaluator import ModelEvaluator


def test_full_pipeline_fast():
    """A fast integration test to ensure components fit together."""
    df = pd.DataFrame(
        {
            "text": ["First mock text", "Second mock text", "Third text", "Fourth one"],
            "target": ["A", "B", "A", "B"],
        }
    )

    # 1. Preprocess
    class MockConfig:
        MAX_FEATURES = 100
        NGRAM_RANGE = (1, 1)
        MIN_DF = 1
        MAX_DF = 1.0

    preprocessor = TextPreprocessor(config=MockConfig())

    X = preprocessor.fit_transform(df["text"])

    # 2. Encode
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["target"])

    # 3. Train
    model = ModelFactory.get_model("MultinomialNB")
    model.train(X, y)

    # 4. Predict
    preds = model.predict(X)
    assert len(preds) == 4

    # 5. Evaluate
    evaluator = ModelEvaluator()
    res = evaluator.evaluate_model(model, X, y, encoder)

    assert res["accuracy"] >= 0.0

    # 6. Save and Load
    model.save("test_model.pkl")
    assert os.path.exists("test_model.pkl")

    loaded = ModelFactory.get_model("MultinomialNB")
    loaded.load("test_model.pkl")
    preds2 = loaded.predict(X)

    assert (preds == preds2).all()
    os.remove("test_model.pkl")
