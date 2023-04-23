import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_greet():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome"


def test_true_positive():
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
    r = client.post("/predict", content=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["0"]["Earn more than 50k"]


def test_true_negative():
    data = {
        "age": [18],
        "workclass": ["Never-worked"],
        "fnlgt": [206359],
        "education": ["8th"],
        "education-num": [4],
        "marital-status": ["Never-married"],
        "occupation": ["?"],
        "relationship": ["Own-child"],
        "race": ["Black"],
        "sex": ["Female"],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [0],
        "native-country": ["Cuba"],
    }
    r = client.post("/predict", content=json.dumps(data))
    assert r.status_code == 200
    assert not r.json()["0"]["Earn more than 50k"]
