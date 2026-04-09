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
            # FORCE pandas to use the basic engine, not the 'arrow' engine that is crashing
            df = pd.read_csv(f, low_memory=False, engine='python')
    print("✅ File Loaded.")
except Exception as e:
    print(f"❌ Load Error: {e}")

# 2. Select Columns
cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Credit_Score']
df = df[cols].copy()

# 3. THE NUCLEAR CLEANER
# We convert to string, remove underscores/junk, and convert to float 
# BEFORE we do any math like '+ 1'
def clean_to_float(value):
    if pd.isna(value): return np.nan
    # Remove everything except numbers and dots
    clean_val = "".join(c for c in str(value) if c.isdigit() or c == '.')
    try:
        return float(clean_val)
    except:
        return np.nan

print("🧹 Scrubbing messy data...")
for col in ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment', 'Interest_Rate']:
    df[col] = df[col].apply(clean_to_float)

# 4. Fill gaps and fix math
df = df.fillna(df.median(numeric_only=True))

# NOW the math will not crash
df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']

# 5. Encoding & Training
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'].astype(str))

X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training AI...")
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

pickle.dump(model, open("credit_model.pkl", "wb"))
print("✅ DONE! Brain created.")
