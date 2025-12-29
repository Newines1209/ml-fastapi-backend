from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Student Performance Predictor API")

# Load models and scaler
logistic_model = joblib.load("logistic_model.joblib")
decision_tree_model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")

# Request body schema
class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Logistic Regression prediction endpoint
@app.post("/predict/logistic")
def predict_logistic(data: Features):
    X = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    X_scaled = scaler.transform(X)
    prediction = logistic_model.predict(X_scaled)
    return {"prediction": int(prediction[0])}

# Decision Tree prediction endpoint
@app.post("/predict/tree")
def predict_tree(data: Features):
    X = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = decision_tree_model.predict(X)
    return {"prediction": int(prediction[0])}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Student Performance Predictor API is running"}
