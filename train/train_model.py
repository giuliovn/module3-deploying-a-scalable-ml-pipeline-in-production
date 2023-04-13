# Script to train machine learning model.
from argparse import ArgumentParser
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger()


def main(data):
    log.info(f"Load and process {data}")
    df = pd.read_csv(data)
    y = LabelEncoder().fit_transform(df["salary"])

    # drop education column (use education-num)
    df = df.drop(columns=["education", "salary"])

    num_features = [
        "age",
        "fnlgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    cat_features = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    transformer = make_column_transformer(
        (OneHotEncoder(), cat_features),
        ("passthrough", num_features),
        remainder="passthrough"
    )
    X = transformer.fit_transform(df)

    log.info("Split data")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20
    )

    log.info("Train Logistic Regression")
    logistic_regression = LogisticRegression(solver="newton-cholesky")
    logistic_regression.fit(X_train, y_train)

    log.info("Evaluate model")
    lr_score = logistic_regression.score(X_val, y_val)
    log.info(f"Score: {lr_score}")

    model_output = Path("model")
    make_dir(model_output)
    log.info(f"Save models to {model_output}")
    joblib.dump(logistic_regression, model_output / "logistic_model.pkl")


def make_dir(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train census model")
    parser.add_argument("data", type=str, help="Path to data")
    args = parser.parse_args()
    main(args.data)
