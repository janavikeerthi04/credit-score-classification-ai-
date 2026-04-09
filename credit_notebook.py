import pandas as pd
import numpy as np
import zipfile
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Data from ZIP
try:
    with zipfile.ZipFile("train.csv.zip", 'r') as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # We use low_memory=False to handle large files better
            df = pd.read_csv(f, low_memory=False)
    print("✅ File Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading file: {e}")

# 2. Select and SCRUB Columns
cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Credit_Score']

df = df[cols].copy()

# This loop finds the columns causing the 'str' error and cleans them
for col in ['Annual_Income', 'Outstanding_Debt', 'Num_of_Delayed_Payment']:
    # Step A: Convert to string
    # Step B: Remove anything that isn't a number or a dot (using Regex)
    # Step C: Convert to a real float number
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

# 3. Fill missing values (NaN) that were created during cleaning
df = df.fillna(df.median(numeric_only=True))

# 4. Feature Engineering (Math will work now!)
df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']

# 5. Encoding Target
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'].astype(str))

# 6. Define Features & Target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Model
print("🚀 Training AI... this will take about 30 seconds...")
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# 9. Save Model
pickle.dump(model, open("credit_model.pkl", "wb"))
print("✅ DONE! 'credit_model.pkl' created. Now run: streamlit run app.py")
