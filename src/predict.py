import joblib
from pathlib import Path

# Absolute path to model file
MODEL_PATH = Path(__file__).resolve().parent.parent / "ml_model" / "breast_cancer_model.pkl"

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    return y_pred