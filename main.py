from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

from train.conf.conf import cat_features
from train.ml.data import process_data
from train.ml.model import inference

app = FastAPI()


def hyphen_to_underscore(string: str):
    return string.replace("-", "_")


class Data(BaseModel):
    age: list[int]
    workclass: list[str]
    fnlgt: list[int]
    education: list[str]
    education_num: list[int]
    marital_status: list[str]
    occupation: list[str]
    relationship: list[str]
    race: list[str]
    sex: list[str]
    capital_gain: list[int]
    capital_loss: list[int]
    hours_per_week: list[int]
    native_country: list[str]

    class Config:
        schema_extra = {
            "example": {
                "age": [31, 18],
                "workclass": ["Private", "Never-worked"],
                "fnlgt": [45781, 206359],
                "education": ["Masters", "8th"],
                "education_num": [14, 4],
                "marital_status": ["Never-married", "Never-married"],
                "occupation": ["Prof-specialty", "?"],
                "relationship": ["Not-in-family", "Own-child"],
                "race": ["White", "Black"],
                "sex": ["Male", "Female"],
                "capital_gain": [14084, 0],
                "capital_loss": [0, 0],
                "hours_per_week": [50, 0],
                "native_country": ["United-States", "Cuba"],
            }
        }


@app.get("/")
async def greet():
    """
    Greets the visitor

    Returns
    -------
    Greeting
    """
    return "Welcome"


@app.post("/predict")
async def predict(
    data: Data,
    model_path="model/model.pkl",
    oh_encoder_path="model/oh_encoder.pkl",
    label_binarizer_path="model/label_binarizer.pkl",
):
    """
    Inputs
    ------
    data : Data object
        Data used for prediction.
    model_path : str
        Path to serialized machine learning model.
    oh_encoder_path: str
        Path to serialized OneHotEncoder
    label_binarizer_path: str
        Path to serialized LabelBinarizer
    Returns
    -------
    preds : dict
        Predictions from the model.
    """
    df = pd.DataFrame(data.dict())
    model = joblib.load(model_path)
    oh_encoder = joblib.load(oh_encoder_path)
    label_binarizer_encoder = joblib.load(label_binarizer_path)
    processed_data, _, _, _ = process_data(
        df,
        categorical_features=list(map(hyphen_to_underscore, cat_features)),
        training=False,
        encoder=oh_encoder,
        lb=label_binarizer_encoder,
    )
    prediction = inference(model, processed_data)
    prediction = [x.item() for x in prediction]  # fastapi doesn't support numpy types
    return {k: {"Earn more than 50k": bool(v)} for v, k in enumerate(prediction)}
