import pandas as pd
import numpy as np
import zipfile
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
try:
    with zipfile.ZipFile("train.csv.zip", 'r') as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # low_memory=False helps with mixed data types
            df = pd.read_csv(f, low_memory=False)
    print("✅ File Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading file: {e}")

# 2. Select Columns
cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Credit_Score']
df = df[cols].copy()

# 3. THE MASTER FIX: Force Text to Numbers
# This removes underscores (_) and any other non-digit characters
clean_cols = ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment', 'Interest_Rate']
for col in clean_cols:
    # We use a 'Regular Expression' to keep only numbers and dots
    df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    # Convert to actual float numbers. 'coerce' turns remaining garbage into NaN (empty)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Handle Missing Values
# Fill any NaNs with the median so the math works
df = df.fillna(df.median(numeric_only=True))

# 5. Feature Engineering (This will NOT fail now!)
df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']

# 6. Encoding Target
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'].astype(str))

# 7. Training the Model
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training AI... please wait about 30 seconds...")
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# 8. Save the 'Brain'
pickle.dump(model, open("credit_model.pkl", "wb"))
print("✅ DONE! 'credit_model.pkl' created. Now run: streamlit run app.py")
