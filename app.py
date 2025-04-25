import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = "loan approval data.csv"
MODEL_PATH = "loan_model.pkl"

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
    }
    .approved {
        background-color: #064e3b;
        border: 1px solid #059669;
        color: #ecfdf5;
    }
    .rejected {
        background-color: #7f1d1d;
        border: 1px solid #dc2626;
        color: #fef2f2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training model on dataset...")
def get_or_train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Loan_Approved"]).copy()
    df["Loan_Approved"] = df["Loan_Approved"].map({"Yes": 1, "No": 0})
    
    drop_columns = ["Applicant_ID", "Loan_Approved"]
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in drop_columns]
    categorical_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    feature_df = pd.get_dummies(df.drop(columns=drop_columns), drop_first=True)
    X = feature_df
    y = df["Loan_Approved"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_df.columns.tolist(),
        "accuracy": acc,
        "auc": auc
    }
    
    joblib.dump(artifact, MODEL_PATH)
    return artifact

artifact = get_or_train_model()
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]

st.title("Bank Loan Approval Predictor")
st.caption("Machine Learning classification pipeline using Logistic Regression.")

col_m1, col_m2 = st.columns(2)
col_m1.metric("Model Test Accuracy", f"{artifact['accuracy'] * 100:.1f}%")
col_m2.metric("ROC-AUC Score", f"{artifact['auc']:.3f}")

st.divider()

st.subheader("Applicant Information")

with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=12000.0, step=500.0)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=2500.0, step=500.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
        dependents = st.number_input("Number of Dependents", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=740.0, step=10.0)
        existing_loans = st.number_input("Existing Loans Count", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
        dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
        savings = st.number_input("Total Savings ($)", min_value=0.0, value=20000.0, step=1000.0)
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])

    with col3:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=500.0, value=25000.0, step=1000.0)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=50000.0, step=1000.0)
        loan_term = st.number_input("Loan Term (Months)", min_value=12.0, max_value=360.0, value=60.0, step=12.0)
        loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Education", "Business", "Car"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Self-Employed"])

    submit_button = st.form_submit_button("Predict Loan Eligibility", type="primary", use_container_width=True)

if submit_button:
    applicant_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category
    }
    
    input_df = pd.DataFrame([applicant_data])
    encoded_df = pd.get_dummies(input_df)
    
    aligned_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in encoded_df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = encoded_df[col].iloc[0]
            
    scaled_input = scaler.transform(aligned_df)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][prediction] * 100
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-box approved">
            Loan Approved (Confidence: {prob:.1f}%)<br>
            <span style="font-size: 14px; font-weight: normal;">Applicant meets the required credit and debt-to-income benchmarks.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box rejected">
            Loan Rejected (Confidence: {prob:.1f}%)<br>
            <span style="font-size: 14px; font-weight: normal;">Risk profile exceeds acceptance threshold based on current financial metrics.</span>
        </div>
        """, unsafe_allow_html=True)
