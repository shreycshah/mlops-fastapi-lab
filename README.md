# FastAPI Breast Cancer Prediction Lab - MLOps

This project demonstrates a simple **ML inference service** built using **FastAPI** and **scikit-learn**.  
It exposes REST APIs for **single and batch predictions**, along with **model metadata** and **dataset metadata** endpoints.

The goal is to showcase a clean, minimal FastAPI + ML setup suitable for demos and learning MLOps fundamentals.

---

## Project Structure

```
mlops-fastapi-lab/
├── assets/
├── ml_model/
│   └── breast_cancer_model.pkl # Trained RandomForestClassifier Base Model
├── src/
│   ├── __init__.py
│   ├── data.py          # Data loading & splitting logic
│   ├── main.py          # FastAPI app and API routes
│   ├── predict.py       # Model loading and inference logic
│   └── train.py         # Model Training and saving logic
├── README.md
└── requirements.txt
```

---

## Model Information

- **Model Type**: RandomForestClassifier  
- **Framework**: scikit-learn  
- **Problem Type**: Binary Classification  
- **Prediction Output**:
  - `0` → Malignant
  - `1` → Benign

---

## Dataset Information

- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: `sklearn.datasets.load_breast_cancer`
- **Features Used**:
  - mean radius
  - mean texture
  - mean smoothness
  - mean perimeter

---

## Setup Instructions

### 1. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server
```bash
uvicorn src.main:app --reload
```

The server will start at:
```
http://127.0.0.1:8000
```

Swagger UI (API docs):
```
http://127.0.0.1:8000/docs
```

---

## Available APIs

### Health Check
```
GET /health
```

### Model Metadata
```
GET /model-info
```

### Dataset Metadata
```
GET /dataset-info
```

### Single Prediction
```
POST /predict
```

### Batch Prediction
```
POST /predict-batch
```
---
This lab is intended for **learning and demonstration purposes**.
