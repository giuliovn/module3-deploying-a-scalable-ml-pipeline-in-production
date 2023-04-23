from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

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
    education_num: list[int] = Field(alias="education-num")
    marital_status: list[str] = Field(alias="marital-status")
    occupation: list[str]
    relationship: list[str]
    race: list[str]
    sex: list[str]
    capital_gain: list[int] = Field(alias="capital-gain")
    capital_loss: list[int] = Field(alias="capital-loss")
    hours_per_week: list[int] = Field(alias="hours-per-week")
    native_country: list[str] = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": [31, 18],
                "workclass": ["Private", "Never-worked"],
                "fnlgt": [45781, 206359],
                "education": ["Masters", "8th"],
                "education-num": [14, 4],
                "marital-status": ["Never-married", "Never-married"],
                "occupation": ["Prof-specialty", "?"],
                "relationship": ["Not-in-family", "Own-child"],
                "race": ["White", "Black"],
                "sex": ["Male", "Female"],
                "capital-gain": [14084, 0],
                "capital-loss": [0, 0],
                "hours-per-week": [50, 0],
                "native-country": ["United-States", "Cuba"],
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
    print(prediction)
    prediction = [x.item() for x in prediction]  # fastapi doesn't support numpy types
    print(prediction)
    print([(v, k) for v, k in enumerate(prediction)])
    return {v: {"Earn more than 50k": bool(k)} for v, k in enumerate(prediction)}
