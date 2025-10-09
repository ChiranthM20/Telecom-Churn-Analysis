import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# -----------------------------
# 1️⃣ Load Environment Variables
# -----------------------------
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# Root connection (no DB selected yet)
root_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306", echo=False)

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
    # Replace spaces in column names
    df.columns = df.columns.str.strip()

    # Replace empty strings with NaN and convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", None), errors='coerce')

    # Fill missing TotalCharges with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("✅ Data transformation complete.")
    return df


# -----------------------------
# 4️⃣ Load to MySQL Database
# -----------------------------
def load_to_db(df):
    try:
        # Create database if not exists
        with root_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
            print(f"✅ Database '{DB_NAME}' checked/created successfully.")

        # Connect to the created database
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}", echo=False)

        # Dimension tables
        dim_customer = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']]
        dim_services = df[['customerID', 'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]
        fact_churn = df[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]

        # Drop existing tables safely
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            conn.execute(text("DROP TABLE IF EXISTS Fact_Churn;"))
            conn.execute(text("DROP TABLE IF EXISTS Dim_Services;"))
            conn.execute(text("DROP TABLE IF EXISTS Dim_Customer;"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()

            # Create Dim_Customer
            conn.execute(text("""
                CREATE TABLE Dim_Customer (
                    customerID VARCHAR(50) PRIMARY KEY,
                    gender VARCHAR(10),
                    SeniorCitizen INT,
                    Partner VARCHAR(10),
                    Dependents VARCHAR(10)
                );
            """))

            # Create Dim_Services
            conn.execute(text("""
                CREATE TABLE Dim_Services (
                    serviceID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(50),
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
                );
            """))

            # Create Fact_Churn
            conn.execute(text("""
                CREATE TABLE Fact_Churn (
                    churnID INT AUTO_INCREMENT PRIMARY KEY,
                    customerID VARCHAR(50),
                    tenure INT,
                    MonthlyCharges FLOAT,
                    TotalCharges FLOAT,
                    Churn VARCHAR(10),
                    FOREIGN KEY (customerID) REFERENCES Dim_Customer(customerID)
                );
            """))
            conn.commit()

        # Load data
        dim_customer.to_sql('Dim_Customer', con=engine, if_exists='append', index=False)
        dim_services.to_sql('Dim_Services', con=engine, if_exists='append', index=False)
        fact_churn.to_sql('Fact_Churn', con=engine, if_exists='append', index=False)

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
