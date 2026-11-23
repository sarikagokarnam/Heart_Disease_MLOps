from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using Machine Learning",
    version="1.0"
)


# Load trained model (absolute path so it works from any working directory)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_disease_model.pkl"
model = joblib.load(MODEL_PATH)


# Input data format
class HeartDiseaseInput(BaseModel):

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = pd.DataFrame([{
        "age": data.age,
        "sex": data.sex,
        "cp": data.cp,
        "trestbps": data.trestbps,
        "chol": data.chol,
        "fbs": data.fbs,
        "restecg": data.restecg,
        "thalach": data.thalach,
        "exang": data.exang,
        "oldpeak": data.oldpeak,
        "slope": data.slope,
        "ca": data.ca,
        "thal": data.thal
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert prediction into readable result
    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return {
        "prediction": int(prediction),
        "result": result
    }