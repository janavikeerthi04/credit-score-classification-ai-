import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model and label encoder
model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

# Friendly labels for Payment History
payment_labels = [
    "Very Poor History", "Poor History", "Average History",
    "Good History", "Very Good History", "Excellent History"
]

# Risk and tips mapping
risk_mapping = {"Poor": "High", "Standard": "Medium", "Good": "Low"}
tips_mapping = {
    "Poor": "Focus on reducing debt and paying bills on time.",
    "Standard": "Maintain good payment habits and monitor debt.",
    "Good": "Keep up good financial habits and low debt."
}

# Page config
st.set_page_config(page_title="Credit Score Predictor", page_icon="💳")

# Title
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

# Feature engineering
debt_ratio = debt / income if income > 0 else 0

features = np.array([[income, debt, payment_history, debt_ratio]], dtype=float)

# Predict Button
if st.button("Predict"):
    try:
        prediction = model.predict(features)[0]
        result = prediction

        risk = risk_mapping[result]
        tips = tips_mapping[result]

        # ---------------- RESULT DISPLAY ----------------
        st.markdown(f"<h2 style='color:green;'>💳 Predicted Credit Score: {result}</h2>", unsafe_allow_html=True)

        if risk == "High":
            st.markdown(f"<h3 style='color:red;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        elif risk == "Medium":
            st.markdown(f"<h3 style='color:orange;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:blue;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='color:#ff00ff; font-size:22px; font-weight:bold;'>💡 Financial Tips: {tips}</p>",
            unsafe_allow_html=True
        )

        # ---------------- METRICS ----------------
        st.subheader("📊 Key Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Income", f"{income}")
        col2.metric("Debt", f"{debt}")
        col3.metric("Debt Ratio", round(debt_ratio, 2))

        # Debt Ratio Indicator
        if debt_ratio > 0.5:
            st.warning("⚠️ High Debt Ratio")
        else:
            st.success("✅ Healthy Debt Ratio")

        # ---------------- BAR CHART ----------------
        st.subheader("📊 Income vs Debt")

        data = pd.DataFrame({
            "Category": ["Income", "Debt"],
            "Amount": [income, debt]
        })

        st.bar_chart(data.set_index("Category"))

        # ---------------- PAYMENT HISTORY PROGRESS ----------------
        st.subheader("📈 Payment History Level")
        progress_value = (payment_history + 1) / len(payment_labels)
        st.progress(progress_value)
        st.write(payment_labels[payment_history])

        # ---------------- PIE CHART (RISK VISUAL) ----------------
        st.subheader("⚠️ Risk Visualization")

        fig, ax = plt.subplots()
        ax.pie([1], labels=[risk])
        st.pyplot(fig)

        # ---------------- SUMMARY ----------------
        st.subheader("📋 Summary")
        st.write(f"Income: {income}")
        st.write(f"Debt: {debt}")
        st.write(f"Debt Ratio: {round(debt_ratio, 2)}")
        st.write(f"Payment History: {payment_labels[payment_history]}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
