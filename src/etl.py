# src/etl.py
import pandas as pd
from sqlalchemy import text
from db import engine, root_engine # Import engine setup from db.py
import os
from dotenv import load_dotenv

# -----------------------------
# 1️⃣ Constants
# -----------------------------
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
    # Start with a copy to avoid SettingWithCopyWarning in subsequent steps
    df_copy = df.copy() 
    
    df_copy.columns = df_copy.columns.str.strip()
    
    # Use .loc and non-inplace assignment for explicit, clean modification (Fixes FutureWarning)
    df_copy.loc[:, 'TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'].replace(" ", None), errors='coerce')
    
        # Resolved: Silencing the Downcasting FutureWarning
    median_charges = df_copy['TotalCharges'].median()
    df_copy.loc[:, 'TotalCharges'] = df_copy['TotalCharges'].fillna(median_charges).astype('float64')
    
    df_copy.drop_duplicates(inplace=True)

    # Add Feature Engineering (Tenure_Group)
    bins = [0, 12, 24, 48, 72]
    labels = ['0-1 year', '1-2 years', '2-4 years', '4-6 years']
    df_copy.loc[:, 'Tenure_Group'] = pd.cut(df_copy['tenure'], bins=bins, labels=labels, right=False).astype(str)

    # Rename Target Column
    df_copy.rename(columns={'Churn': 'Churn_Label'}, inplace=True)
    
    print("✅ Data transformation complete.")
    return df_copy


# -----------------------------
# 4️⃣ Load to MySQL Database
# -----------------------------
def load_to_db(df):
    try:
        # Create database if not exists using the root connection
        with root_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
            conn.commit()
            print(f"✅ Database '{DB_NAME}' checked/created successfully.")

        # Create DataFrame views/slices using .copy() to avoid SettingWithCopyWarning
        dim_customer = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']].copy()
        dim_services = df[['customerID', 'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']].copy()
        fact_customeractivity = df[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                                     'Churn_Label', 'Tenure_Group', 'PaymentMethod']].copy() 

        # Corrected Step: Ensure customerID is correctly truncated/typed 
        # Use .loc for explicit assignment to suppress SettingWithCopyWarning
        dim_customer.loc[:, 'customerID'] = dim_customer['customerID'].astype(str).str[:20]
        dim_services.loc[:, 'customerID'] = dim_services['customerID'].astype(str).str[:20]
        fact_customeractivity.loc[:, 'customerID'] = fact_customeractivity['customerID'].astype(str).str[:20]

        # Use a single 'begin' transaction for all DDL (Schema changes)
        with engine.begin() as conn: 
            
            # Step 1: Disable and Drop Tables
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            # Use CONSISTENT LOWERCASE table names in DDL
            conn.execute(text("DROP TABLE IF EXISTS fact_customeractivity;")) 
            conn.execute(text("DROP TABLE IF EXISTS dim_services;"))
            conn.execute(text("DROP TABLE IF EXISTS dim_customer;"))

            # Step 2: Create Tables (using LOWERCASE names)
            conn.execute(text("""
                CREATE TABLE dim_customer (
                    customerID VARCHAR(20) PRIMARY KEY,
                    gender VARCHAR(10),
                    SeniorCitizen INT,
                    Partner VARCHAR(10),
                    Dependents VARCHAR(10)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))

            # Create dim_services (using LOWERCASE names)
            conn.execute(text("""
                CREATE TABLE dim_services (
                    serviceID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(20) NOT NULL,
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
                    FOREIGN KEY (customerID) REFERENCES dim_customer(customerID)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))

            # Create fact_customeractivity (using LOWERCASE names)
            conn.execute(text("""
                CREATE TABLE fact_customeractivity (
                    churnID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(20) NOT NULL,
                    tenure INT,
                    MonthlyCharges FLOAT,
                    TotalCharges FLOAT,
                    Churn_Label VARCHAR(10),
                    Tenure_Group VARCHAR(20),
                    PaymentMethod VARCHAR(50),
                    FOREIGN KEY (customerID) REFERENCES dim_customer(customerID)
                ) ENGINE=InnoDB CHARACTER SET utf8mb4;
            """))
            
            # Step 3: Re-enable Foreign Key Checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;")) 

        # Step 4: Load Data 
        # Use CONSISTENT LOWERCASE table names in to_sql() (Fixes UserWarning)
        dim_customer.to_sql('dim_customer', con=engine, if_exists='append', index=False)
        dim_services.to_sql('dim_services', con=engine, if_exists='append', index=False)
        fact_customeractivity.to_sql('fact_customeractivity', con=engine, if_exists='append', index=False)

        print("✅ Data successfully loaded into MySQL tables!")

    except Exception as e:
        print(f"❌ DB error: {e}")


# -----------------------------
# 5️⃣ Main ETL Flow
# -----------------------------
if __name__ == "__main__":
    df = extract_data()
    df = transform_data(df)
    load_to_db(df)