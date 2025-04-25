# Loan Approval Prediction System

An end-to-end Machine Learning pipeline that predicts bank loan approval using Logistic Regression and Scikit-Learn.

## Project Overview
Financial institutions need fast, consistent, and data-driven methods to evaluate loan applications. This project trains and evaluates an interpretable classification pipeline to predict whether a customer qualifies for a loan based on credit history, income, collateral, debt-to-income (DTI) ratio, and applicant demographics.

## Features
- Data Preprocessing: Handles missing values using median (numerical features) and mode (categorical features).
- Feature Scaling: Applies standard scaling (`StandardScaler`) to normalize feature magnitudes.
- Model Training: Trains a Logistic Regression classifier with stratified train-test splitting.
- Evaluation Metrics: Generates Test Accuracy, ROC-AUC score, and precision-recall classification reports.
- Inference Pipeline: Includes a single-applicant prediction function returning approval status and confidence percentage.

## Technologies Used
- Python 3.10+
- Scikit-Learn
- Pandas
- NumPy
- Joblib

## Project Structure
```text
loan-Approval-system/
├── app.py                  # Training pipeline, evaluation, and inference script
├── loan approval data.csv  # Historical applicant dataset
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/HarshwardhanSingh145/loan-Approval-system.git
cd loan-Approval-system
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
Execute the complete training and inference workflow:
```bash
python3 app.py
```

Sample output:
```text
Model Training Complete
Test Accuracy: 86.84%
ROC-AUC Score: 0.9190

Classification Report:
              precision    recall  f1-score   support
    Rejected       0.89      0.92      0.91       130
    Approved       0.82      0.75      0.78        60

Result: Loan Approved (Confidence: 94.5%)
```

