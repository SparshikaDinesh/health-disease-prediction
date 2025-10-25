import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown

# ✅ Step 2: Use this exact gdown format
url = "https://drive.google.com/uc?id=1qeA73w0m9wzx8FaJIivf54ikKML32O_M"
output = "best_model_rf.joblib"
gdown.download(url, output, quiet=False)

# Load model and scaler
model = joblib.load('best_model_rf.joblib')
scaler = joblib.load('scaler.joblib')

st.title("🩺 Diabetes Risk Predictor (Demo)")
st.write("This app predicts the probability of diabetes based on key health indicators.")

# Collect user inputs
st.header("Enter your health details")

inputs = []
features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

for feat in features:
    val = st.number_input(feat, value=0.0, step=1.0)
    inputs.append(val)

# Make prediction
if st.button("🔍 Predict"):
    x = np.array([inputs]).astype(float)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0, 1]
    st.success(f"Predicted probability of diabetes: **{prob:.3f}**")
