from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Student Performance ML API")

# Load models
logistic_model = joblib.load("logistic_model.joblib")
decision_tree_model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")

# Input schema
class StudentInput(BaseModel):
    study_hours: float
    attendance: float

@app.get("/")
def home():
    return {"message": "Student Performance ML API is running"}

@app.post("/predict/logistic")
def predict_logistic(data: StudentInput):
    features = np.array([[data.study_hours, data.attendance]])
    features_scaled = scaler.transform(features)
    prediction = logistic_model.predict(features)
    return {"prediction": int(prediction[0])}

@app.post("/predict/decision-tree")
def predict_decision_tree(data: StudentInput):
    features = np.array([[data.study_hours, data.attendance]])
    prediction = decision_tree_model.predict(features)
    return {"prediction": int(prediction[0])}
