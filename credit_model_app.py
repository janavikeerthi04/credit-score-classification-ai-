import streamlit as st
import numpy as np
import joblib

# Load model & encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")  # Payment_Behaviour encoder

st.title("💳 Credit Score Predictor")

# Inputs
income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Outstanding Debt", min_value=0.0)
payment = st.selectbox("Payment Behaviour", le.classes_)  # dropdown shows strings

# Predict button
if st.button("Predict"):
    # Encode Payment_Behaviour to integer
    payment_encoded = le.transform([payment])[0]

    # Calculate debt ratio
    debt_ratio = debt / income if income != 0 else 0

    # Create feature array in correct order
    features = np.array([[income, debt, payment_encoded, debt_ratio]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Map prediction to readable category
    if prediction == 0:
        result = "Poor"
    elif prediction == 1:
        result = "Standard"
    else:
        result = "Good"

    st.success(f"Your Credit Score Category: {result}")
