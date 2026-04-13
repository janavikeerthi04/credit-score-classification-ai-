# 💳 Credit Score Predictor

A Machine Learning web application built using **Streamlit** that predicts a user's **credit score category** based on financial inputs.
_______________________________________________________________________________________________________________________
## 🚀 Project Overview

This application uses a trained **Random Forest Classifier** to classify individuals into:

- Poor
- Standard
- Good
based on their financial behavior.

---

## 📊 Features

- User-friendly web interface (Streamlit)
- Predicts credit score category instantly
- Displays:
  - Credit Score Category
  - Risk Level (High / Medium / Low)
  - Financial Improvement Tips
- Handles edge cases (like zero income)
- Clean UI with styled outputs

---

## 🧠 Machine Learning Model

- Algorithm: Random Forest Classifier
- Features used:
  - Annual Income
  - Outstanding Debt
  - Payment History
  - Debt-to-Income Ratio (calculated)

---

## 🧾 Input Parameters

| Input | Description |
|------|------------|
| Annual Income | User's yearly income |
| Outstanding Debt | Total current debt |
| Payment History | Financial discipline level (Very Poor → Excellent) |

---

## 📈 Output

- **Credit Score Category** → Poor / Standard / Good  
- **Risk Level** → High / Medium / Low  
- **Financial Tips** → Suggestions to improve financial health  

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- NumPy
- Joblib

---

## 📂 Project Structure
credit-score-classification-ai/ │ ├── credit_model.pkl        # Trained ML model ├── credit_model_app.py     # Streamlit application ├── requirements.txt        # Dependencies └── README.md               # Project documentation
---

## 🌐 Deployment

This app can be deployed using:

- Streamlit Cloud

---

## 💡 Example Use Case

| Income | Debt | Payment History | Output |
|-------|------|----------------|--------|
| 20000 | 15000 | Poor | Poor |
| 50000 | 20000 | Average | Standard |
| 100000 | 10000 | Excellent | Good |

---

## 🎯 Future Improvements

- Add loan approval prediction
- Add graphical dashboard
- Improve UI design
- Add explainable AI (feature importance)

---

## 👩‍💻 Author

Developed by **janavi**

---

## ❤️ Acknowledgment

This project was built as part of an AI/ML learning journey and showcases end-to-end model development and deployment.
