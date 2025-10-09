# src/api.py
"""
Minimal FastAPI wrapper to get churn prediction for a single record.
This expects the same preprocessing pipeline used in training.
Run with:
uvicorn src.api:app --reload --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_DIR = "models"
meta = joblib.load(os.path.join(MODEL_DIR, "meta.joblib"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
encoders = {}
for c in meta.get('cat_cols', []):
    path = os.path.join(MODEL_DIR, f"enc_{c}.joblib")
    if os.path.exists(path):
        encoders[c] = joblib.load(path)

app = FastAPI(title="Churn Predictor API")

class CustomerRecord(BaseModel):
    # Accept a flexible dict shape; common fields:
    customerID: str = None
    tenure: float = 0
    MonthlyCharges: float = 0.0
    TotalCharges: float = 0.0
    Tenure_Group: str = None
    PaymentMethod: str = None

@app.post("/predict")
def predict(record: CustomerRecord):
    rec = record.dict()
    # Build DataFrame with required features order
    X = pd.DataFrame([{}])
    for c in meta['feature_columns']:
        X[c] = [rec.get(c, 0)]

    # apply encoders
    for c, le in encoders.items():
        if c in X.columns:
            val = str(X.at[0, c]) if X.at[0, c] is not None else "NA"
            if val not in set(le.classes_):
                val = "NA"
            try:
                X[c] = le.transform([val])[0]
            except Exception:
                X[c] = 0

    # scale numeric
    num_cols = meta.get('num_cols', [])
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])

    prob = float(rf.predict_proba(X)[0,1])
    pred = int(rf.predict(X)[0])
    return {"churn_prob": prob, "churn_pred": pred}
