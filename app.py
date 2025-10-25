import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown

# Download model from Google Drive
url = "https://drive.google.com/uc?id=1qeA73w0m9wzx8FaJIivf54ikKML32O_M"
gdown.download(url, "best_model_rf.joblib", quiet=False)

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
    val = st.number_in_
