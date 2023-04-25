from argparse import ArgumentParser
import json

import requests

data = {
    "age": [31],
    "workclass": ["Private"],
    "fnlgt": [45781],
    "education": ["Masters"],
    "education-num": [14],
    "marital-status": ["Never-married"],
    "occupation": ["Prof-specialty"],
    "relationship": ["Not-in-family"],
    "race": ["White"],
    "sex": ["Male"],
    "capital-gain": [14084],
    "capital-loss": [0],
    "hours-per-week": [50],
    "native-country": ["United-States"],
}


def main(endpoint):
    response = requests.post(endpoint, data=json.dumps(data))
    print(f"Return code: {response.status_code}")
    print(f"Return json: {response.json()}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Live API test")

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        help="API endpoint",
        default="https://ud-giuliovn-predict-income-c96k.onrender.com/predict",
    )

    args = parser.parse_args()

    main(args.endpoint)
