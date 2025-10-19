# app/main.py
from fastapi import FastAPI
from app.schemas import HeartData
import joblib
import numpy as np

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease using a trained ML model",
    version="1.0.0"
)

# Load model
model = joblib.load("model/heart_model.joblib")

# Root route
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running fine."}

@app.get("/info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "input_features": [
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalach", "exang",
            "oldpeak", "slope", "ca", "thal"
        ],
        "output": "heart_disease (0 = No, 1 = Yes)"
    }

@app.post("/predict")
def predict(data: HeartData):
    # Convert input to numpy array
    features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                          data.fbs, data.restecg, data.thalach, data.exang,
                          data.oldpeak, data.slope, data.ca, data.thal]])
    
    prediction = model.predict(features)[0]
    result = bool(prediction)

    return {
        "heart_disease": result,
        "message": "Positive for heart disease" if result else "No heart disease detected"
    }
 