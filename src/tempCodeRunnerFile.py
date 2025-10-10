# src/etl.py
import pandas as pd
from sqlalchemy import text
from db import engine, root_engine # Import engine setup from db.py

# -----------------------------
# 1️⃣ Constants
# -----------------------------
# Import DB_NAME from environment (or fetch from db.py if needed, but safer to rely on env)
import os
from dotenv import load_dotenv
load_dotenv()
DB_NAME = os.getenv("DB_NAME", "telco_dw") 


# -----------------------------
# 2️⃣ Extract Data
# -----------------------------
def extract_data():
    df = pd.read_csv("data/raw/telco_churn.csv")
    print(f"✅ Data extracted: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -----------------------------
# 3️⃣ Transform Data
# -----------------------------
def transform_data(df):
    df.columns = df.columns.str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", None), errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop_duplicates(inplace=True)

    # CRITICAL FIX 1: Add Feature Engineering (Tenure_Group) for ML scripts
    bins = [0, 12, 24, 48, 72]
    labels = ['0-1 year', '1-2 years', '2-4 years', '4-6 years']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False).astype(str)

    # CRITICAL FIX 2: Rename Target Column for ML consistency
    df.rename(columns={'Churn': 'Churn_Label'}, inplace=True)
    
    print("✅ Data transformation complete.")
    return df


# Updated section 4️⃣ Load to MySQL Database in src/etl.py
def load_to_db(df):
    try:
        # Create database if not exists using the root connection
        with root_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
            conn.commit()
            print(f"✅ Database '{DB_NAME}' checked/created successfully.")

        # Dimension and Fact tables... (DataFrame creation remains the same)
        dim_customer = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']]
        dim_services = df[['customerID', 'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]
        fact_customeractivity = df[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                                    'Churn_Label', 'Tenure_Group', 'PaymentMethod']] 

        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            conn.execute(text("DROP TABLE IF EXISTS Fact_CustomerActivity;")) 
            conn.execute(text("DROP TABLE IF EXISTS Dim_Services;"))
            conn.execute(text("DROP TABLE IF EXISTS Dim_Customer;"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()

            # Create Dim_Customer - ADDED ENGINE/CHARSET
            conn.execute(text("""
                CREATE TABLE Dim_Customer (
                    customerID VARCHAR(50) PRIMARY KEY,
                    gender VARCHAR(10),
                    SeniorCitizen INT,
                    Partner VARCHAR(10),
                    Dependents VARCHAR(10)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))
            conn.commit() # Commit Dim_Customer immediately

            # Create Dim_Services - ADDED ENGINE/CHARSET
            conn.execute(text("""
                CREATE TABLE Dim_Services (
                    serviceID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(50) NOT NULL,
                    PhoneService VARCHAR(10),
                    MultipleLines VARCHAR(20),
                    InternetService VARCHAR(30),
                    OnlineSecurity VARCHAR(20),
                    OnlineBackup VARCHAR(20),
                    DeviceProtection VARCHAR(20),
                    TechSupport VARCHAR(20),
                    StreamingTV VARCHAR(20),
                    StreamingMovies VARCHAR(20),
                    Contract VARCHAR(20),
                    PaperlessBilling VARCHAR(10),
                    PaymentMethod VARCHAR(50),
                    FOREIGN KEY (customerID) REFERENCES Dim_Customer(customerID)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))

            # Create Fact_CustomerActivity - ADDED ENGINE/CHARSET
            conn.execute(text("""
                CREATE TABLE Fact_CustomerActivity (
                    churnID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(50) NOT NULL,
                    tenure INT,
                    MonthlyCharges FLOAT,
                    TotalCharges FLOAT,
                    Churn_Label VARCHAR(10),
                    Tenure_Group VARCHAR(20),
                    PaymentMethod VARCHAR(50),
                    FOREIGN KEY (customerID) REFERENCES Dim_Customer(customerID)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))
            conn.commit()

        # Load data (Unchanged)
        dim_customer.to_sql('Dim_Customer', con=engine, if_exists='append', index=False)
        dim_services.to_sql('Dim_Services', con=engine, if_exists='append', index=False)
        fact_customeractivity.to_sql('Fact_CustomerActivity', con=engine, if_exists='append', index=False)

        print("✅ Data successfully loaded into MySQL tables!")

    except Exception as e:
        # Re-raise the error to see the full traceback if it persists
        print(f"❌ DB error: {e}")


# -----------------------------
# 5️⃣ Main ETL Flow
# -----------------------------
if __name__ == "__main__":
    df = extract_data()
    df = transform_data(df)
    load_to_db(df)