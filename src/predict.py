# src/predict.py
"""
Load saved model + preprocessors, compute churn probabilities for all customers,
and write churn_prob & churn_pred back to Fact_CustomerActivity (DB).
"""
import os
import joblib
import pandas as pd
from db import engine

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
    df = pd.read_sql("SELECT * FROM Fact_CustomerActivity", engine)
    return df

def prepare_X(df, meta, encoders):
    # prepare feature matrix consistent with training
    X = pd.DataFrame()
    for c in meta['feature_columns']:
        if c in df.columns:
            X[c] = df[c]
        else:
            # missing column -> fill default
            X[c] = 0

    # numeric inverse transform if scaler expects numeric standardization
    num_cols = meta.get('num_cols', [])
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    # apply label encoders
    for c, le in encoders.items():
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("NA")
            # handle unseen labels - map to a nearest known label index (use transform with try/except)
            known = set(le.classes_)
            X[c] = X[c].apply(lambda val: val if val in known else "NA")
            try:
                X[c] = le.transform(X[c])
            except Exception:
                # fallback: produce zeros if transform fails
                X[c] = 0

    # scale numeric using scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])

    return X

def write_predictions(df_with_preds):
    # Only keep the fields we need to update in FACT table
    # Using to_sql replace the table; to update in-place you'd use SQLAlchemy core/upsert.
    # Here we'll replace table for simplicity but keep columns that already exist.
    temp_table = "Fact_CustomerActivity"
    df_up = df_with_preds.copy()
    # Keep all original columns plus churn_prob/churn_pred
    df_up.to_sql(temp_table, engine, if_exists='replace', index=False)
    print(f"Predictions written to table {temp_table} (replaced).")

def main():
    rf, scaler, meta, encoders = load_stuff()
    df_fact = load_fact()
    if df_fact.empty:
        raise SystemExit("Fact table empty. Run ETL first.")

    X = prepare_X(df_fact, meta, encoders)
    # compute probs/pred
    probs = rf.predict_proba(X)[:, 1]
    preds = rf.predict(X)

    df_fact['churn_prob'] = probs
    df_fact['churn_pred'] = preds.astype(int)

    # write back
    write_predictions(df_fact)
    print("Prediction complete. churn_prob and churn_pred columns updated in Fact_CustomerActivity.")

if __name__ == "__main__":
    main()
