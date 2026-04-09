import pandas as pd
import numpy as np
import zipfile
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# 1. Load Data
try:
    with zipfile.ZipFile("train.csv.zip", 'r') as z:
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    print("✅ Data loaded from ZIP!")
except Exception as e:
    print(f"❌ Load error: {e}")

# 2. Select columns
cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Credit_Score']
df = df[cols].copy()

# 3. AGGRESSIVE CLEANING ENGINE
# This function rips out everything except numbers and dots
def hardcore_clean(text):
    text = str(text)
    # Remove everything that isn't 0-9 or a period .
    clean_text = re.sub(r'[^0-9.]', '', text)
    if clean_text == '' or clean_text == '.':
        return np.nan
    return float(clean_text)

print("🧹 Scrubbing junk characters (underscores, letters, etc.)...")
target_cols = ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment', 'Interest_Rate', 'Num_of_Loan']

for col in target_cols:
    df[col] = df[col].apply(hardcore_clean)

# 4. Fill gaps (crucial for XGBoost)
df = df.fillna(df.median(numeric_only=True))

# 5. Feature Engineering
# Now math is safe because everything is a float!
df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']

# 6. Encode Label
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'].astype(str))

# 7. Train
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training AI model...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# 8. Save
pickle.dump(model, open("credit_model.pkl", "wb"))
print("✅ DONE! 'credit_model.pkl' created. Now run: streamlit run app.py")
