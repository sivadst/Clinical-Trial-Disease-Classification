import pytest
import numpy as np
from src.models.classifiers import SklearnClassifierWrapper, ModelFactory
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from sklearn.dummy import DummyClassifier


@pytest.fixture
def mock_config():
    class MockConfig:
        RANDOM_STATE = 42
        CLASSIFIERS = ["MultinomialNB"]
        N_FOLDS = 2

    return MockConfig()


@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_sklearn_wrapper(dummy_data):
    X, y = dummy_data
    model = DummyClassifier(strategy="most_frequent")
    wrapper = SklearnClassifierWrapper(model, "Dummy")

    wrapper.train(X, y)
    preds = wrapper.predict(X)
    assert len(preds) == 100

    probs = wrapper.predict_proba(X)
    assert probs.shape == (100, 2)

    metrics = wrapper.evaluate(X, y)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics


def test_model_factory():
    nb = ModelFactory.get_model("MultinomialNB")
    assert nb.model_name == "MultinomialNB"

    rf = ModelFactory.get_model("RandomForest", n_estimators=10)
    assert rf.model.n_estimators == 10


def test_trainer_cross_validate(mock_config, dummy_data):
    X, y = dummy_data
    trainer = ModelTrainer(config=mock_config)

    model = ModelFactory.get_model("MultinomialNB")

    results = trainer.cross_validate(model, X, y)
    assert "accuracy_mean" in results
    assert "f1_mean" in results


def test_evaluator(dummy_data):
    X, y = dummy_data
    model = DummyClassifier(strategy="most_frequent")
    wrapper = SklearnClassifierWrapper(model, "Dummy")
    wrapper.train(X, y)

    class MockLabelEncoder:
        classes_ = np.array(["class0", "class1"])

        def inverse_transform(self, arr):
            return np.array([self.classes_[i] for i in arr])

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(wrapper, X, y, MockLabelEncoder())

    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results
