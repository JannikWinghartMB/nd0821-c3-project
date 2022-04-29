from fastapi.testclient import TestClient

from main import app, PersonData

client = TestClient(app)


def test_get_path():
    # Act
    r = client.get("/")

    # Assert
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome."}


def test_post_inference_case_0():
    # Arange
    body = {
        "age": 39,
        "workclass": 'State-gov',
        "fnlwgt": 77516,
        "education": 'Bachelors',
        "education-num": 13,
        "marital-status": 'Never-married',
        "occupation": 'Adm-clerical',
        "relationship": 'Not-in-family',
        "race": 'White',
        "sex": 'Male',
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": 'United-States'
    }


    # Act
    r = client.post("/", json=body)

    # Assert
    assert r.status_code == 200
    assert r.json() == {
        "Prediction": 0,
        "Prediction_text": "<=$50K/yr"
    }


def test_post_inference_case_1():
    # Arange
    body = {
        "age": 58,
        "workclass": 'Federal-gov',
        "fnlwgt": 200042,
        "education": 'Some-college',
        "education-num": 10,
        "marital-status": 'Married-civ-spouse',
        "occupation": 'Exec-managerial',
        "relationship": 'Husband',
        "race": 'White',
        "sex": 'Male',
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": 'United-States'
    }

    # Act
    r = client.post("/", json=body)

    # Assert
    assert r.status_code == 200
    assert r.json() == {
        "Prediction": 1,
        "Prediction_text": ">$50K/yr"
    }