import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Car Price Prediction API")

model = joblib.load("models/model_v1.pkl")

MODEL_COLUMNS = ["km_driven", "fuel", "seller_type", "transmission", "owner",
                 "mileage",	"engine", "max_power", "seats",	"brand", "car_age"]


class CarInput(BaseModel):
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float
    brand: str
    car_age: int


@app.get("/")
def read_root():
    return {
        "message": "Car Price Prediction API is running",
        "docs": "Visit /docs to try the prediction endpoint interactively"
    }


@app.post("/predict")
def predict(data: CarInput):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict], columns=MODEL_COLUMNS)

    prediction = model.predict(input_df)
    actual_price = np.expm1(prediction[0])

    return {"predicted_price": round(float(actual_price), 2)}


