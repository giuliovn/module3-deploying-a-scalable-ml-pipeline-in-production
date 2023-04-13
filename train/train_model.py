from argparse import ArgumentParser
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from conf.conf import cat_features
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger()


def main(data):
    log.info(f"Load and process {data}")
    df = pd.read_csv(data)
    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, oh_encoder, label_binarizer = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        encoder=oh_encoder,
        lb=label_binarizer,
        training=False,
    )

    log.info("Train")
    model = train_model(X_train, y_train)

    log.info("Predict on test")
    y_pred = inference(model, X_test)

    log.info("Evaluate")
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    log.info(f"Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}")

    model_output = Path("model")
    model_output.mkdir(parents=True, exist_ok=True)
    model_path = model_output / "model.pkl"
    log.info(f"Save models to {model_path}")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train census model")
    parser.add_argument("data", type=str, help="Path to data")
    args = parser.parse_args()
    main(args.data)
