import streamlit as st
import numpy as np
import joblib

model = joblib.load("credit_model.pkl")
le = joblib.load("label_encoder.pkl")

st.write("Model expects:", model.n_features_in_)

st.title("💳 Credit Score Predictor")

income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Outstanding Debt", min_value=0.0)

payment_behavior = st.selectbox(
    "Payment Behaviour",
    ["Low_spent", "High_spent", "Average"]  # adjust based on dataset
)

if st.button("Predict"):

    # Encode
    payment_encoded = le.transform([payment_behavior])[0]

    # Feature engineering (same as training)
    debt_ratio = debt / income if income != 0 else 0

    # ✅ NOW 4 FEATURES (matches training)
    features = np.array([[income, debt, payment_encoded, debt_ratio]])

    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Poor"
    elif prediction[0] == 1:
        result = "Standard"
    else:
        result = "Good"

    st.success(f"Your Credit Score Category: {result}")
