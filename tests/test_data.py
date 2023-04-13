import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from train.conf.conf import cat_features


def test_cat_features_type(data):
    for feature in cat_features:
        assert (data[feature] == type(object)).all


def test_num_features_type(data):
    for feature in set(data.columns) - set(cat_features):
        assert (data[feature] == type(np.number)).all


def test_process_data(train):
    assert len(train["X_train"].shape) == 2
    assert len(train["y_train"].shape) == 1
    assert isinstance(train["oh_encoder"], OneHotEncoder)
    assert isinstance(train["label_binarizer"], LabelBinarizer)
