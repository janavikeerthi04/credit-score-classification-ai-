import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("credit_model.pkl")

# Debug: check how many features model expects
st.write("Model expects:", model.n_features_in_)

st.title("💳 Credit Score Predictor")
st.write("Enter your financial details:")

# Inputs
income = st.number_input("Annual Income", min_value=0.0)
debt = st.number_input("Outstanding Debt", min_value=0.0)
payment_history = st.slider("Payment History Score", 0, 100)

# Prediction button
if st.button("Predict"):
    # ✅ Use ONLY 3 features (most likely correct)
    features = np.array([[income, debt, payment_history]])

    try:
        prediction = model.predict(features)

        if prediction[0] == 0:
            result = "Poor"
        elif prediction[0] == 1:
            result = "Standard"
        else:
            result = "Good"

        st.success(f"Your Credit Score Category: {result}")

    except Exception as e:
        st.error(f"Error: {e}")
