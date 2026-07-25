import pytest
import pandas as pd
from src.data.splitter import DataSplitter


@pytest.fixture
def splitter():
    class MockConfig:
        TARGET_COLUMN = "target"
        TEST_SIZE = 0.2
        VAL_SIZE = 0.25  # Of remaining 80%, so 20% total
        RANDOM_STATE = 42

    return DataSplitter(config=MockConfig())


@pytest.fixture
def dummy_data():
    return pd.DataFrame({"feature1": range(100), "target": ["classA"] * 50 + ["classB"] * 50})


def test_split_stratification(splitter, dummy_data):
    train, val, test = splitter.split(dummy_data)

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20

    # Check stratification
    train_dist = train["target"].value_counts(normalize=True).to_dict()
    val_dist = val["target"].value_counts(normalize=True).to_dict()
    test_dist = test["target"].value_counts(normalize=True).to_dict()

    assert train_dist["classA"] == 0.5
    assert train_dist["classB"] == 0.5

    assert val_dist["classA"] == 0.5
    assert val_dist["classB"] == 0.5

    assert test_dist["classA"] == 0.5
    assert test_dist["classB"] == 0.5


def test_get_split_statistics(splitter, dummy_data):
    train, val, test = splitter.split(dummy_data)
    stats = splitter.get_split_statistics(train, val, test)

    assert stats["train"]["classA"] == 0.5
    assert stats["val"]["classA"] == 0.5
    assert stats["test"]["classA"] == 0.5
