import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

# Friendly names for Payment Behaviour
payment_labels = ["Very Bad", "Bad", "Average", "Good", "Very Good", "Excellent"]

st.title("💳 Credit Score Predictor")
st.write("Enter your financial details:")

# Inputs
income = st.number_input("Annual Income", min_value=0.0, step=100.0)
debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)

# Select Payment Behaviour in words
payment_index = st.select_slider(
    "Payment Behaviour",
    options=list(range(len(payment_labels))),
    format_func=lambda x: payment_labels[x]
)

# Calculate debt_ratio
debt_ratio = debt / income if income > 0 else 0

# Prepare features for prediction
features = np.array([[income, debt, payment_index, debt_ratio]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)[0]

    # Map prediction to friendly category
    category_mapping = {0: "Poor", 1: "Standard", 2: "Good"}
    risk_mapping = {0: "High", 1: "Medium", 2: "Low"}
    tips_mapping = {
        0: "Focus on reducing debt and paying bills on time.",
        1: "Maintain good payment habits and monitor debt.",
        2: "Keep up good financial habits and low debt."
    }

    prediction = int(model.predict(features)[0])
    result = category_mapping[prediction]
    risk = risk_mapping[prediction]
    tips = tips_mapping[prediction]

    st.success(f"Predicted Credit Score Category: {result}")
    st.warning(f"Risk Level: {risk}")
    st.info(f"💡 Financial Tips: {tips}")
