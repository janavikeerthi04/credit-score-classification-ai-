import streamlit as st
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

# Friendly labels for Payment Behaviour
payment_labels = ["Very Bad", "Bad", "Average", "Good", "Very Good", "Excellent"]

# Risk and tips mapping
risk_mapping = {"Poor": "High", "Standard": "Medium", "Good": "Low"}
tips_mapping = {
    "Poor": "Focus on reducing debt and paying bills on time.",
    "Standard": "Maintain good payment habits and monitor debt.",
    "Good": "Keep up good financial habits and low debt."
}

st.title("💳 Credit Score Predictor")
st.write("Enter your financial details:")

# User Inputs
income = st.number_input("Annual Income", min_value=0.0, step=100.0)
debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)

# Payment Behaviour slider with friendly labels
payment_index = st.select_slider(
    "Payment Behaviour",
    options=list(range(len(payment_labels))),
    format_func=lambda x: payment_labels[x]
)

# Calculate debt_ratio safely
debt_ratio = debt / income if income > 0 else 0

# Prepare features as 2D array
features = np.array([[income, debt, payment_index, debt_ratio]], dtype=float)

# Prediction button
if st.button("Predict"):
    try:
        # Model prediction returns string category
        prediction = model.predict(features)[0]
        result = prediction

        risk = risk_mapping[result]
        tips = tips_mapping[result]

        # Colored outputs using markdown
        st.markdown(f"<h2 style='color:green;'>💳 Predicted Credit Score: {result}</h2>", unsafe_allow_html=True)

        if risk == "High":
            st.markdown(f"<h3 style='color:red;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        elif risk == "Medium":
            st.markdown(f"<h3 style='color:orange;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:blue;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)

        st.markdown(f"<p style='color:purple;'>💡 Financial Tips: {tips}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
