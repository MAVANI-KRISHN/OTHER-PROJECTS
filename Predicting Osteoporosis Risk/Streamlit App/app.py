# Streamlit App Code Run Statement 'python -m streamlit run "Streamlit App/app.py"'

import streamlit as st
import numpy as np
import pandas as pd
import joblib

label_encoders = joblib.load("Code/Label Encoder/label_encoders.pkl")
scaler = joblib.load("Code/Standard Scaler/scaler_model.pkl")

rf_model = joblib.load("Code/Without GridSearchCV Model/rf_model.pkl")
gscv_rf_best = joblib.load("Code/With GridSearchCV Model/gscv_rf_best_model.pkl") 

dt_model = joblib.load("Code/Without GridSearchCV Model/dt_model.pkl")
gscv_dt_best = joblib.load("Code/With GridSearchCV Model/gscv_dt_best_model.pkl") 

lr_model = joblib.load("Code/Without GridSearchCV Model/lr_model.pkl")
gscv_lr_best = joblib.load("Code/With GridSearchCV Model/gscv_lr_best_model.pkl") 

svc_model = joblib.load("Code/Without GridSearchCV Model/svc_model.pkl")
gscv_svc_best = joblib.load("Code/With GridSearchCV Model/gscv_svc_best_model.pkl")  

def predict_osteoporosis_risk(patient_data, models):
    processed_data = []
    
    for col in patient_data:
        if col in label_encoders:
            try:
                processed_value = label_encoders[col].transform([patient_data[col]])[0]
            except ValueError:
                return f"Invalid value '{patient_data[col]}' for {col}."
        else:
            processed_value = float(patient_data[col])
        processed_data.append(processed_value)

    processed_data = np.array(processed_data).reshape(1, -1)
    processed_data = scaler.transform(processed_data)
    
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(processed_data)[0]
        predictions[model_name] = "High Risk" if prediction == 1 else "Low Risk"
    
    return predictions

st.title("Osteoporosis Risk Prediction")
st.write("Enter patient details to predict osteoporosis risk.")

age = st.slider("Age", min_value=0, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
hormonal_changes = st.selectbox("Hormonal Changes", ["Normal", "Postmenopausal"])
family_history = st.selectbox("Family History", ["Yes", "No"])
race = st.selectbox("Race/Ethnicity", ["Asian", "Caucasian", "African American"])
body_weight = st.selectbox("Body Weight", ["Underweight", "Normal"])
calcium_intake = st.selectbox("Calcium Intake", ["Low", "Adequate"])
vitamin_d = st.selectbox("Vitamin D Intake", ["Sufficient", "Insufficient"])
physical_activity = st.selectbox("Physical Activity", ["Sedentary", "Active"])
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ["Moderate", "None"])
medical_conditions = st.selectbox("Medical Conditions", ["Rheumatoid Arthritis", "None", "Hyperthyroidism"])
medications = st.selectbox("Medications", ["Corticosteroids", "None"])
prior_fractures = st.selectbox("Prior Fractures", ["Yes", "No"])

patient_data = {
    'Age': age,
    'Gender': gender,
    'Hormonal Changes': hormonal_changes,
    'Family History': family_history,
    'Race/Ethnicity': race,
    'Body Weight': body_weight,
    'Calcium Intake': calcium_intake,
    'Vitamin D Intake': vitamin_d,
    'Physical Activity': physical_activity,
    'Smoking': smoking,
    'Alcohol Consumption': alcohol,
    'Medical Conditions': medical_conditions,
    'Medications': medications,
    'Prior Fractures': prior_fractures
}

if st.button("Predict Risk"):
    models = {
        "Random Forest": rf_model,
        "GridSearchCV Random Forest": gscv_rf_best,
        "Decision Tree": dt_model,
        "GridSearchCV Decision Tree": gscv_dt_best,
        "Logistic Regression": lr_model,
        "GridSearchCV Logistic Regression": gscv_lr_best,
        "SVC": svc_model,
        "GridSearchCV SVC": gscv_svc_best
    }

    result = predict_osteoporosis_risk(patient_data, models)

    without_gscv_results = {
        model_name: result[model_name] 
        for model_name in result if "GridSearchCV" not in model_name
    }

    with_gscv_results = {
        model_name: result[model_name] 
        for model_name in result if "GridSearchCV" in model_name
    }

    model_order = ["Random Forest", "Decision Tree", "Logistic Regression", "SVC"]

    result_df = pd.DataFrame({
        'Model Name': model_order,
        'Without GridSearchCV Prediction': [without_gscv_results[model] for model in model_order],
        'With GridSearchCV Prediction': [with_gscv_results[f"GridSearchCV {model}"] for model in model_order]
    })

    st.dataframe(result_df)