import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and label encoder
model = joblib.load("bond_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load SHAP explainer (optional)
explainer = joblib.load("shap_explainer.pkl")

# Page title
st.title("📊 Municipal Bond Credit Rating Predictor")
st.write("Enter bond features to predict its credit rating and interpret with SHAP.")

# Input form
coupon_rate = st.number_input("Coupon (%)", min_value=0.0, max_value=20.0, value=5.0)
bond_yield = st.number_input("Yield (%)", min_value=0.0, max_value=20.0, value=4.5)
price = st.number_input("Price ($)", min_value=50.0, max_value=150.0, value=100.0)
duration = st.number_input("Duration (Years)", min_value=0.0, max_value=50.0, value=10.0)

# Prepare input for prediction
input_data = pd.DataFrame({
    "Coupon": [coupon_rate],
    "Yield": [bond_yield],
    "Price": [price],
    "Duration": [duration]
})

# Predict and show result
if st.button("Predict Credit Rating"):
    prediction_encoded = model.predict(input_data)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"🧾 Predicted Credit Rating: **{prediction}**")

    # SHAP explanation
    st.subheader("🔍 SHAP Explanation")
    shap_values = explainer.shap_values(input_data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(explainer.expected_value[prediction_encoded], 
                    shap_values[prediction_encoded][0, :], 
                    input_data.iloc[0], matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')
