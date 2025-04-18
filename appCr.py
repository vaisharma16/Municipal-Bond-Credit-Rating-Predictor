import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import shap
st.write(f"SHAP Version being used: {shap.__version__}")
# Streamlit SHAP rendering function
from streamlit.components.v1 import html
html("<h1>Hello from HTML!</h1>")

def st_shap(plot, height=None):
    from streamlit.components.v1 import html
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    html(shap_html, height=height or 400, scrolling=True)

# Load model and encoder
try:
    model = joblib.load("bond_rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    explainer = joblib.load("shap_explainer.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model or related files: {e}")
    st.stop()

# Streamlit Page
st.title("📊 Municipal Bond Credit Rating Predictor")
st.write("Enter bond features to predict its credit rating and interpret with SHAP.")

# Input fields
coupon_rate = st.number_input("Coupon (%)", min_value=0.0, max_value=20.0, value=4.25)
bond_yield = st.number_input("Yield (%)", min_value=0.0, max_value=20.0, value=3.20)
price = st.number_input("Price ($)", min_value=50.0, max_value=150.0, value=101.50)
duration = st.number_input("Duration (Years)", min_value=0.0, max_value=50.0, value=7.10)

# Create input DataFrame
input_df = pd.DataFrame({
    "Coupon": [coupon_rate],
    "Yield": [bond_yield],
    "Price": [price],
    "Duration": [duration]
})

# Predict
if st.button("Predict Credit Rating"):
    prediction_encoded = model.predict(input_df)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    st.success(f"🧾 Predicted Credit Rating: **{prediction}**")


    # Get SHAP values
    shap_values = explainer.shap_values(input_df)

    st.subheader("🔍 SHAP Explanation (Summary Plot with st_shap Test)")
    shap_values_summary = explainer.shap_values(input_df)
    if isinstance(shap_values_summary, list):
        shap.summary_plot(shap_values_summary[prediction_encoded], input_df, show=False)
    else:
        shap.summary_plot(shap_values_summary, input_df, show=False)
    import matplotlib.pyplot as plt
    st.pyplot(plt.gcf()) # Use st.pyplot for matplotlib plots