from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
logistic_model = joblib.load("logistic_model.joblib")
decision_tree_model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")

# MODEL EXPECTS ONLY 2 FEATURES
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    X = np.array([[data.feature1, data.feature2]])
    X_scaled = scaler.transform(X)
    prediction = logistic_model.predict(X_scaled)
    return {
        "model": "logistic_regression",
        "prediction": int(prediction[0])
    }

@app.post("/predict/tree")
def predict_tree(data: InputData):
    X = np.array([[data.feature1, data.feature2]])
    X_scaled = scaler.transform(X)
    prediction = decision_tree_model.predict(X_scaled)
    return {
        "model": "decision_tree",
        "prediction": int(prediction[0])
    }

@app.get("/")
def root():
    return {"message": "API is working"}
