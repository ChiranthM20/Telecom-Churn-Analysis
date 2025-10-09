# src/train.py
"""
Train models:
- Loads processed fact table from MySQL
- Performs simple preprocessing, train-test split
- Trains Logistic Regression and RandomForest
- Saves models, encoders, scaler to models/
"""
import os
import joblib
import pandas as pd
import numpy as np
from db import engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_fact():
    query = "SELECT * FROM Fact_CustomerActivity"
    df = pd.read_sql(query, engine)
    return df

def prepare_features(df):
    # Keep Customer_ID for later mapping
    df = df.copy()
    # Drop rows missing churn label
    if 'Churn_Label' in df.columns:
        df = df[df['Churn_Label'].notnull()]
    else:
        df['Churn_Label'] = 0

    # Select features
    # numeric candidates
    num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in df.columns]
    # categorical candidates
    cat_cols = [c for c in ['Tenure_Group','PaymentMethod'] if c in df.columns]

    features = num_cols + cat_cols
    X = df[features].copy()
    y = df['Churn_Label'].astype(int)

    # Fill numeric missing with median
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X[c].fillna(X[c].median(), inplace=True)

    encoders = {}
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le
        joblib.dump(le, os.path.join(MODEL_DIR, f"enc_{c}.joblib"))

    # scaler for numeric
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    return X, y, encoders, num_cols, cat_cols, df[['Customer_ID']]

def train_and_save(X_train, y_train):
    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logreg.joblib"))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf.joblib"))

    return lr, rf

def evaluate(model, X_test, y_test, name="model"):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        probs = None

    print(f"--- Evaluation for {name} ---")
    print(classification_report(y_test, preds, digits=4))
    if probs is not None:
        print("AUC:", roc_auc_score(y_test, probs))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def main():
    df = load_fact()
    if df.empty:
        raise SystemExit("Fact table is empty. Run ETL first.")
    X, y, encoders, num_cols, cat_cols, id_df = prepare_features(df)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, id_df, test_size=0.2, random_state=42, stratify=y)

    lr, rf = train_and_save(X_train, y_train)
    evaluate(lr, X_test, y_test, name="Logistic Regression")
    evaluate(rf, X_test, y_test, name="Random Forest")

    # Save column list for inference
    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_columns": list(X.columns)
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.joblib"))
    print("Training complete. Models saved to models/")

if __name__ == "__main__":
    main()
