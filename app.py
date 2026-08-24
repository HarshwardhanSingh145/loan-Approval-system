import gradio as gr
import pandas as pd
from pycaret.classification import load_model, predict_model

# 1. Model load karein
model = load_model('loan_model')

# 2. Data load karein (Sirf UI ke dabe banane ke liye)
# Note: 'loan approval data.csv' aapke actual file ka naam hona chahiye
df = pd.read_csv("jupyterinstall/loan approval data.csv") 
df = df.drop(columns=['Loan_Approved', 'application_id']) 
input_columns = df.columns

def predict_loan(*user_inputs):
    input_dict = {col: [val] for col, val in zip(input_columns, user_inputs)}
    input_df = pd.DataFrame(input_dict)
    prediction = predict_model(model, data=input_df)
    result = prediction['prediction_label'].iloc[0]
    return "✅ Congratulations! Loan Approved." if result == 1 else "❌ Sorry! Loan Rejected."

ui_inputs = []
for col in input_columns:
    if df[col].dtype == 'object':
        ui_inputs.append(gr.Dropdown(choices=df[col].dropna().unique().tolist(), label=col))
    else:
        ui_inputs.append(gr.Number(label=col))

demo = gr.Interface(fn=predict_loan, inputs=ui_inputs, outputs="text", title="🏦 Smart Bank Loan Predictor")
demo.launch()
