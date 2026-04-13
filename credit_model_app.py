import streamlit as st
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

# Friendly labels for Payment History
payment_labels = ["Very Poor History", "Poor History", "Average History", "Good History", "Very Good History", "Excellent History"]

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

payment_history = st.select_slider(
    "Payment History",
    options=list(range(len(payment_labels))),
    format_func=lambda x: payment_labels[x]
)

# Calculate debt_ratio safely
debt_ratio = debt / income if income > 0 else 0

features = np.array([[income, debt, payment_history, debt_ratio]], dtype=float)

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

        # Big, bright financial tips
        st.markdown(
            f"<p style='color:#ff00ff; font-size:22px; font-weight:bold;'>💡 Financial Tips: {tips}</p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
