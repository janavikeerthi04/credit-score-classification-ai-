import streamlit as st
import numpy as np
import joblib
import pandas as pd

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
        # Model prediction
        prediction = model.predict(features)[0]
        result = prediction

        risk = risk_mapping[result]
        tips = tips_mapping[result]

        # Result display
        st.markdown(f"<h2 style='color:green;'>💳 Predicted Credit Score: {result}</h2>", unsafe_allow_html=True)

        if risk == "High":
            st.markdown(f"<h3 style='color:red;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        elif risk == "Medium":
            st.markdown(f"<h3 style='color:orange;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:blue;'>⚠️ Risk Level: {risk}</h3>", unsafe_allow_html=True)

        # Financial tips
        st.markdown(
            f"<p style='color:#ff00ff; font-size:22px; font-weight:bold;'>💡 Financial Tips: {tips}</p>",
            unsafe_allow_html=True
        )

        # ---------------- REAL EXPLAINABLE AI ----------------
        st.subheader("🧠 Why this prediction? (Model-Based Explainable AI)")

        # Feature names (must match training order)
        feature_names = ["Income", "Debt", "Payment History", "Debt Ratio"]

        # Get feature importance from model
        importances = model.feature_importances_

        # Create dataframe
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # Show importance graph
        st.bar_chart(importance_df.set_index("Feature"))

        # Show top 2 important features
        st.write("🔍 Top factors affecting your score:")

        top_features = importance_df.head(2)

        for _, row in top_features.iterrows():
            st.write(f"👉 {row['Feature']} has high impact on prediction")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
