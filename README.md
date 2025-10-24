# health-disease-prediction
Health prediction project using machine learning — predicts diabetes risk from health indicators (BMI, Age, Blood Pressure, etc.) with Random Forest and SHAP explainability.
# 🩺 Health Disease Prediction

Health prediction project using **machine learning** — predicts **diabetes risk** from health indicators (BMI, Age, Blood Pressure, etc.) using **Random Forest** and **SHAP explainability**.

---

## 🚀 Project Overview
- Cleaned and analyzed the CDC BRFSS 2015 health dataset  
- Trained multiple models: Logistic Regression, Random Forest, XGBoost  
- Evaluated model performance (Accuracy, ROC AUC, Confusion Matrix)  
- Used SHAP for model explainability (to understand feature importance)  
- Deployed a simple Streamlit app for interactive predictions  

---

## 🧠 Key Insights
- **Top predictors:** BMI, Age, Blood Pressure, Cholesterol  
- **Best model:** Random Forest (AUC ≈ 0.84)  
- SHAP visualizations show how health indicators affect diabetes likelihood  

---

## ⚙️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
