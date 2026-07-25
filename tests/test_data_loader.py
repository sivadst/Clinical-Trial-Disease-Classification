import pytest
import pandas as pd
from unittest.mock import patch

from src.data.loader import DataLoader
from src.utils.exceptions import DataValidationError


@pytest.fixture
def mock_config():
    class MockConfig:
        RAW_DATA_PATH = "dummy/path.csv"
        TARGET_COLUMN = "target"
        TEXT_COLUMN = "text"

    return MockConfig()


@pytest.fixture
def loader(mock_config):
    return DataLoader(config=mock_config)


@pytest.fixture
def valid_df():
    return pd.DataFrame(
        {"target": ["class1", "class2", "class1"], "text": ["text1", "text2", "text3"]}
    )


def test_load_valid_data(loader, valid_df):
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=valid_df):
            df = loader.load_raw_data()
            assert df.shape == (3, 2)
            assert list(df.columns) == ["target", "text"]


def test_load_missing_file(loader):
    with patch("src.data.loader.Path.exists", return_value=False):
        with pytest.raises(DataValidationError, match="Data file not found"):
            loader.load_raw_data()


def test_load_empty_dataframe(loader):
    empty_df = pd.DataFrame()
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=empty_df):
            with pytest.raises(DataValidationError, match="empty"):
                loader.load_raw_data()


def test_missing_required_columns(loader):
    invalid_df = pd.DataFrame({"other_col": [1, 2, 3]})
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=invalid_df):
            with pytest.raises(DataValidationError, match="Missing required columns"):
                loader.load_raw_data()


def test_single_class_target(loader):
    invalid_df = pd.DataFrame(
        {"target": ["class1", "class1", "class1"], "text": ["text1", "text2", "text3"]}
    )
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=invalid_df):
            with pytest.raises(DataValidationError, match="least 2 classes"):
                loader.load_raw_data()


def test_all_null_target(loader):
    invalid_df = pd.DataFrame({"target": [None, None, None], "text": ["text1", "text2", "text3"]})
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=invalid_df):
            with pytest.raises(DataValidationError, match="all nulls"):
                loader.load_raw_data()


def test_all_null_text(loader):
    invalid_df = pd.DataFrame(
        {"target": ["class1", "class2", "class3"], "text": [None, None, None]}
    )
    with patch("src.data.loader.Path.exists", return_value=True):
        with patch("src.data.loader.pd.read_csv", return_value=invalid_df):
            with pytest.raises(DataValidationError, match="all nulls"):
                loader.load_raw_data()


def test_get_target_distribution(loader, valid_df):
    dist = loader.get_target_distribution(valid_df)
    assert dist == {"class1": 2, "class2": 1}


def test_get_data_summary(loader, valid_df):
    summary = loader.get_data_summary(valid_df)
    assert summary["shape"] == (3, 2)
    assert summary["target_distribution"] == {"class1": 2, "class2": 1}
    assert summary["duplicates"] == 0
