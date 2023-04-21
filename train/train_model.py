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
        test,
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

    log.info("Evaluate overall performance")
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

    log.info(f"Precision: {precision}. Recall: {recall}. Fbeta: {fbeta}")

    log.info("Check performance on data slices")
    reindex_test = test.reset_index()
    slice_df = pd.DataFrame(
        columns=["Feature", "Value", "Count", "Precision", "Recall", "Fbeta"]
    )
    for feature in cat_features:
        for feature_value in reindex_test[feature].unique():
            log.debug(f"Feature: {feature}. Value: {feature_value}")
            slice_index = reindex_test[reindex_test[feature] == feature_value].index
            slice_precision, slice_recall, slice_fbeta = compute_model_metrics(
                y_test[slice_index], y_pred[slice_index]
            )
            log.debug(
                f"Precision: {slice_precision}. Recall: {slice_recall}. Fbeta: {slice_fbeta}"
            )
            slice_df = slice_df._append(
                {
                    "Feature": feature,
                    "Value": feature_value,
                    "Count": len(slice_index),
                    "Precision": slice_precision,
                    "Recall": slice_recall,
                    "Fbeta": slice_fbeta,
                },
                ignore_index=True,
            )

    model_output = Path("model")
    model_output.mkdir(parents=True, exist_ok=True)

    log.info(f"Save slice performance in {model_output} directory")
    slice_df.to_csv(model_output / "slice_performance.csv")
    slice_df.to_html(model_output / "slice_performance.html")

    log.info(f"Save model and encoders to {model_output}")
    joblib.dump(model, model_output / "model.pkl")
    joblib.dump(oh_encoder, model_output / "oh_encoder.pkl")
    joblib.dump(label_binarizer, model_output / "label_binarizer.pkl")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train census model")
    parser.add_argument("data", type=str, help="Path to data")
    args = parser.parse_args()
    main(args.data)
