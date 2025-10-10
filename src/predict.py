# src/predict.py
"""
Load saved model + preprocessors, compute churn probabilities for all customers,
and write churn_prob & churn_pred back to Fact_CustomerActivity (DB).
"""
import os
import joblib
import pandas as pd
from db import engine # Import engine from db.py

MODEL_DIR = "models"

def load_stuff():
    rf = joblib.load(os.path.join(MODEL_DIR, "rf.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.joblib"))
    encoders = {}
    for c in meta.get('cat_cols', []):
        enc_path = os.path.join(MODEL_DIR, f"enc_{c}.joblib")
        if os.path.exists(enc_path):
            encoders[c] = joblib.load(enc_path)
    return rf, scaler, meta, encoders

def load_fact():
    # Corrected table name
    df = pd.read_sql("SELECT * FROM Fact_CustomerActivity", engine)
    return df

def prepare_X(df, meta, encoders, scaler):
    # prepare feature matrix consistent with training
    X = pd.DataFrame()
    for c in meta['feature_columns']:
        if c in df.columns:
            X[c] = df[c]
        else:
            X[c] = 0

    num_cols = meta.get('num_cols', [])
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    # apply label encoders
    for c, le in encoders.items():
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("NA")
            known = set(le.classes_)
            X[c] = X[c].apply(lambda val: val if val in known else "NA")
            try:
                X[c] = le.transform(X[c])
            except Exception:
                X[c] = 0

    # scale numeric using scaler
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])

    return X

def write_predictions(df_with_preds):
    # Overwrite the original Fact table with the new columns added (churn_prob, churn_pred)
    df_with_preds.to_sql('Fact_CustomerActivity', engine, if_exists='replace', index=False)
    print("Predictions written to table Fact_CustomerActivity (replaced).")

def main():
    rf, scaler, meta, encoders = load_stuff()
    df_fact = load_fact()
    if df_fact.empty:
        raise SystemExit("Fact table empty. Run ETL (src/etl.py) first.")

    X = prepare_X(df_fact, meta, encoders, scaler)
    
    probs = rf.predict_proba(X)[:, 1]
    preds = rf.predict(X)

    df_fact['churn_prob'] = probs
    df_fact['churn_pred'] = preds.astype(int)

    write_predictions(df_fact)
    print("Prediction complete. churn_prob and churn_pred columns updated in Fact_CustomerActivity.")

if __name__ == "__main__":
    main()