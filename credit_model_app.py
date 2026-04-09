import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

# Map encoded Payment_Behaviour back to original labels
payment_classes = list(le.classes_)  # e.g., ['Poor', 'Below Average', 'Average', 'Good', 'Very Good', 'Excellent']
payment_mapping = {i: cls for i, cls in enumerate(payment_classes)}

st.title("💳 Credit Score Predictor")
st.write("Enter your financial details:")

# Inputs
income = st.number_input("Annual Income", min_value=0.0, step=100.0)
debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)

# Slider for Payment Behaviour (show user-friendly names)
payment_index = st.select_slider(
    "Payment Behaviour",
    options=list(range(len(payment_classes))),
    format_func=lambda x: payment_mapping[x]
)

# Calculate debt_ratio
debt_ratio = debt / income if income > 0 else 0

# Prepare features for model
features = np.array([[income, debt, payment_index, debt_ratio]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]

    # Map prediction to category names (0=Poor, 1=Standard, 2=Good, adjust if different)
    if prediction == 0:
        result = "Poor"
        risk = "High"
        tips = "Focus on reducing debt and paying bills on time."
    elif prediction == 1:
        result = "Standard"
        risk = "Medium"
        tips = "Maintain good payment habits and monitor debt."
    else:
        result = "Good"
        risk = "Low"
        tips = "Keep up good financial habits and low debt."

    st.success(f"Predicted Credit Score Category: {result}")
    st.warning(f"Risk Level: {risk}")
    st.info(f"💡 Financial Tips: {tips}")
