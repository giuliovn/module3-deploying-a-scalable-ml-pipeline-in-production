import pytest
from sklearn.ensemble import AdaBoostClassifier

from train.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope="module", autouse=True)
def model(train):
    return train_model(train["X_train"], train["y_train"])


def test_train_model(model):
    assert isinstance(model, AdaBoostClassifier)


@pytest.fixture(scope="module", autouse=True)
def prediction(model, train):
    return inference(model, train["X_train"])


def test_inference(prediction):
    assert len(prediction.shape) == 1


def test_compute_model_metrics(train, prediction):
    precision, recall, fbeta = compute_model_metrics(train["y_train"], prediction)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
