import pandas as pd
import numpy as np
import os
import zipfile
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- STEP 1 & 2: LOADING THE ZIP FILE ---
file_name = "train.csv.zip"

try:
    with zipfile.ZipFile(file_name, 'r') as z:
        csv_name = z.namelist()[0] 
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    print(f"✅ SUCCESS: Loaded {csv_name}!")
except Exception as e:
    print(f"❌ ERROR: Check if '{file_name}' is in your folder. Details: {e}")

# --- STEP 4 & 5: PREPROCESSING & CLEANING ---
try:
    # We clean the data specifically for this classification task
    cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
            'Credit_Utilization_Ratio', 'Credit_Score']
    
    df = df[cols].copy()
    
    # Handle missing values (Numerical with median, Categorical with mode)
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode the target (Credit_Score)
    le = LabelEncoder()
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'].astype(str))
    
    # Feature Engineering (The ratios your Sir requested)
    df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']
    
    print("✅ SUCCESS: Data Preprocessing complete!")
except Exception as e:
    print(f"❌ ERROR in Preprocessing: {e}")

# --- STEP 6 & 7: FEATURES & SPLIT ---
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 8 - 12: MODEL & SAVE ---
print("🚀 Training model... please wait...")
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Save the brain
pickle.dump(model, open("credit_model.pkl", "wb"))
print("✅ SUCCESS: 'credit_model.pkl' has been created!")
