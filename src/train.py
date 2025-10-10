# src/train.py
import os
import joblib
import pandas as pd
from db import engine # Import engine from db.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_fact():
    # Corrected Fact table name
    query = "SELECT * FROM Fact_CustomerActivity"
    df = pd.read_sql(query, engine)
    return df

def prepare_features(df):
    df = df.copy()

    # 1. Prepare Target Variable (Y)
    # Map 'Yes'/'No' from DB to 1/0
    df['Churn_Target'] = df['Churn_Label'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    y = df['Churn_Target']
    
    # 2. Select features
    num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in df.columns]
    cat_cols = [c for c in ['Tenure_Group','PaymentMethod'] if c in df.columns]

    features = num_cols + cat_cols
    X = df[features].copy()

    # CRITICAL FIX 7: Use correct customer ID column name
    id_df = df[['customerID']] 

    # 3. Fill numeric missing with median
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X[c].fillna(X[c].median(), inplace=True)

    # 4. Apply Label Encoders
    encoders = {}
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
        le = LabelEncoder()
        le.fit(X[c])
        X[c] = le.transform(X[c])
        encoders[c] = le
        joblib.dump(le, os.path.join(MODEL_DIR, f"enc_{c}.joblib"))
    
    # 5. Scale numeric features
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    return X, y, encoders, num_cols, cat_cols, id_df

def train_and_save(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logreg.joblib"))

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf.joblib"))

    return lr, rf

def evaluate(model, X_test, y_test, name="model"):
    # ... (Evaluation logic remains the same) ...
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
        raise SystemExit("Fact table is empty. Run ETL (src/etl.py) first.")
        
    X, y, encoders, num_cols, cat_cols, id_df = prepare_features(df)
    
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, id_df, test_size=0.2, random_state=42, stratify=y)

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