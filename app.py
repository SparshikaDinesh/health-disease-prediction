
import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.title("Diabetes Risk Predictor (Demo)")

model = joblib.load('best_model_rf.joblib')
scaler = joblib.load('scaler.joblib')

st.write("Enter values for the features:")

# Create inputs dynamically (simple number_input for all)
inputs = []
for feat in ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']:
    val = st.number_input(feat, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    x = np.array([inputs]).astype(float)
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0,1]
    st.write(f"Predicted probability of diabetes: {prob:.3f}")
