# src/db.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "telco_dw")

CONN_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# echo=True can be set for SQLAlchemy debugging
engine = create_engine(CONN_STR, pool_recycle=3600)
