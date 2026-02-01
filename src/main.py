from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from src.predict import predict_data
import numpy as np

app = FastAPI()

class BreastCancerData(BaseModel):
    mean_radius : float
    mean_texture : float
    mean_smoothness : float
    mean_perimeter : float

class BreastCancerResponse(BaseModel):
    response : float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=BreastCancerResponse)
async def predict_breast_cancer(breast_cancer_features: BreastCancerData):
    try:
        features = np.array([[
            breast_cancer_features.mean_radius,
            breast_cancer_features.mean_texture,
            breast_cancer_features.mean_smoothness,
            breast_cancer_features.mean_perimeter
        ]])

        prediction = predict_data(features)
        return BreastCancerResponse(response=int(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_name": "Breast Cancer Random Forest Classifier",
        "model_type": "RandomForestClassifier",
        "framework": "scikit-learn",
        "model_version": "1.0.0",
        "inference_mode": "batch_and_single"
    }

@app.get("/dataset-info")
def dataset_info():
    return {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "sklearn.datasets.load_breast_cancer",
        "problem_type": "Binary Classification",
        "feature_schema": {
            "mean_radius": "float",
            "mean_texture": "float",
            "mean_smoothness": "float",
            "mean_perimeter": "float"
        },
        "target_definition": {
            "0": "malignant",
            "1": "benign"
        },
    }