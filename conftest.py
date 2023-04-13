from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from train.conf.conf import cat_features
from train.ml.data import process_data

test_data = Path(__file__).parent / "data" / "census_clean.csv"


@pytest.fixture(scope="session", autouse=True)
def data():
    return pd.read_csv(test_data)


@pytest.fixture(scope="session", autouse=True)
def train(data):
    train, _ = train_test_split(data, test_size=0.20)
    X_train, y_train, oh_encoder, label_binarizer = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "oh_encoder": oh_encoder,
        "label_binarizer": label_binarizer,
    }
