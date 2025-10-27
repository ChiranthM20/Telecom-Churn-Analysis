import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------- Page Setup -------------------
st.set_page_config(page_title="Telecom Customer Churn Analysis", layout="wide")
st.title("📞 Telecom Customer Churn Analysis Dashboard")

st.sidebar.title("About this App")
st.sidebar.info("""
- 📂 Upload your Telecom dataset (CSV)  
- 🧹 The app will clean and encode your data automatically  
- 🤖 Trains Logistic Regression and Random Forest models  
- 📊 Displays Accuracy, Reports, and Visual Graphs  
""")

st.sidebar.markdown(
    """
    <div style="text-align:center; margin-top:20px;">
        <b>🧠 Developed by <span style="color:#00BFFF;">Srujan R</span> and his Team</b><br>
        <small>Mini Project | Data Warehousing & Data Mining</small>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------- Upload Section -------------------
uploaded_file = st.file_uploader("📤 Upload your Telecom Churn Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head())

    st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

    # ------------------- Data Preprocessing -------------------
    st.subheader("🧹 Data Preprocessing")

    data = data.dropna()
    label_enc = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = label_enc.fit_transform(data[col].astype(str))

    target_col = st.selectbox("🎯 Select Target Column (Churn)", data.columns, index=len(data.columns)-1)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success("✅ Data preprocessed and split into training and testing sets!")

    # ------------------- Logistic Regression -------------------
    st.subheader("🤖 Logistic Regression Model")
    log_model = LogisticRegression(max_iter=500)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)

    acc_log = accuracy_score(y_test, y_pred_log)
    st.write(f"Accuracy: {acc_log:.2%}")
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title("Logistic Regression - Confusion Matrix")
    st.pyplot(fig)

    # ------------------- Random Forest -------------------
    st.subheader("🌳 Random Forest Model")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Accuracy: {acc_rf:.2%}")
    st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))

    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=ax2)
    plt.title("Random Forest - Confusion Matrix")
    st.pyplot(fig2)

    # ------------------- Visualization -------------------
    st.subheader("📊 Data Insights")

    # Churn distribution
    fig3, ax3 = plt.subplots()
    sns.countplot(x=target_col, data=data, palette="coolwarm", ax=ax3)
    plt.title("Customer Churn Distribution")
    st.pyplot(fig3)

    # Correlation heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=False, cmap='viridis', ax=ax4)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig4)

    # Feature importance (for Random Forest)
    st.subheader("🌟 Important Features Affecting Churn")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(importances.set_index('Feature'))

    st.success("✅ Analysis Completed Successfully!")

else:
    st.warning("👆 Please upload a Telecom Churn dataset to begin analysis.")