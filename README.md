# Telecom-Churn-Analysis

# Telecom Customer Churn Analysis

Folder layout:
- data/raw/ : put `telco_churn.csv` here (Kaggle WA_Fn-UseC_-Telco-Customer-Churn.csv)
- data/processed/ : outputs from ETL
- sql/ : SQL DDL to create DW
- src/ : python scripts (etl.py, train.py, predict.py, api.py)
- models/ : saved model artifacts

Quick run:
1. Create DB: run `sql/telco_dw_ddl.sql` in MySQL Workbench.
2. Copy `.env.example` → `.env` and set DB credentials.
3. Install deps: `pip install -r requirements.txt`
4. Put dataset CSV in `data/raw/telco_churn.csv`
5. Run ETL: `python src/etl.py`
6. Run training: `python src/train.py`
7. Run prediction & save to DB: `python src/predict.py`
8. Optional: run API: `uvicorn src.api:app --reload --port 8000`

Notes:
- Scripts use SQLAlchemy + pymysql to connect to MySQL.
- Save label encoders/scaler/models are stored into `models/`.
