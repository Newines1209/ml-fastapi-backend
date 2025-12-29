from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load models and scaler
logistic_model = joblib.load("logistic_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")
scaler = joblib.load("scaler.joblib")

app = FastAPI()

# Add CORS so your frontend can access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class StudentData(BaseModel):
    study_hours: float
    attendance: float

# Logistic regression endpoint
@app.post("/predict/logistic")
def predict_logistic(data: StudentData):
    features = np.array([[data.study_hours, data.attendance]])
    features_scaled = scaler.transform(features)
    prediction = logistic_model.predict(features_scaled)
    return {"prediction": int(prediction[0])}

# Decision tree endpoint
@app.post("/predict/tree")
def predict_tree(data: StudentData):
    features = np.array([[data.study_hours, data.attendance]])
    features_scaled = scaler.transform(features)
    prediction = tree_model.predict(features_scaled)
    return {"prediction": int(prediction[0])}
