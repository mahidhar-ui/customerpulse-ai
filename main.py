from fastapi import FastAPI
import sys

sys.path.append("src")

from predict import predict_customer


app = FastAPI(
    title="CustomerPulse AI API",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "CustomerPulse AI API is running"
    }


@app.post("/predict")
def predict(data: dict):

    result = predict_customer(
        data
    )

    return result