import os
import sys
import subprocess
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import load_model, inference
from starter.ml.data import process_data

app = FastAPI()

# source: https://knowledge.udacity.com/questions/689224
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    ls_output = subprocess.run(["ls", "-a"], capture_output=True, text=True)
    print(ls_output.stdout)
    print(ls_output.stderr)
    ls_output = subprocess.run(["ls", "-a", "/app"], capture_output=True, text=True)
    print(ls_output.stdout)
    print(ls_output.stderr)
    os.system("rm -f -r .dvc/tmp/lock .dvc/cache")
    os.system("dvc config core.no_scm true")
    dvc_output = subprocess.run(["dvc", "pull"], capture_output=True, text=True)
    if dvc_output.returncode != 0:
        print(dvc_output.stdout)
        print(dvc_output.stderr)
        print("dvc pull failed")
    else:
        print("dvc pull successful")
    os.system("rm -r .dvc .apt/usr/lib/dvc -f")


class PersonData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }

@app.get("/")
async def welcome():
    return {"message": "Welcome."}


@app.post("/")
async def perform_inference(input_person: PersonData):
    model, encoder, lb = load_model("model/trained_model")
    df = pd.DataFrame([input_person.dict()])
    df = df.rename(columns={
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country"
    })
    df["salary"] = 0

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = inference(model, X).tolist()[0]
    return {
        "Prediction": prediction,
        "Prediction_text": "<=$50K/yr" if prediction==0 else ">$50K/yr"
    }
