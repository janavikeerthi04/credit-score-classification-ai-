import streamlit as st
import numpy as np
import joblib

# -------------------------
# Load trained objects
# -------------------------
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")  # Payment_Behaviour encoder

st.set_page_config(page_title="💳 Credit Score Predictor", layout="centered")
st.title("💳 Credit Score Predictor")
st.write("Enter your financial details to predict your credit score and risk level.")

# -------------------------
# User Inputs
# -------------------------
income = st.number_input("Annual Income", min_value=0.0, step=1000.0, format="%.2f")
debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0, format="%.2f")
payment = st.selectbox("Payment Behaviour", le.classes_)  # dropdown with original categories

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    # Encode Payment_Behaviour
    payment_encoded = le.transform([payment])[0]

    # Calculate debt_ratio
    debt_ratio = debt / income if income != 0 else 0

    # Prepare feature array
    features = np.array([[income, debt, payment_encoded, debt_ratio]])

    # Predict
    pred = model.predict(features)[0]

    # Map prediction to category
    if pred == 0:
        category = "Poor"
        risk = "High"
        tips = "Reduce debt and pay bills on time."
    elif pred == 1:
        category = "Standard"
        risk = "Medium"
        tips = "Maintain regular payments and keep debt low."
    else:
        category = "Good"
        risk = "Low"
        tips = "Keep up good financial habits and low debt."

    # Display results
    st.success(f"✅ Predicted Credit Score Category: {category}")
    st.info(f"⚠️ Risk Level: {risk}")
    st.write(f"💡 Financial Tips: {tips}")
