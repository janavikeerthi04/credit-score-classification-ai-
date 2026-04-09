import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
# Updated to read from a zip file directly
df = pd.read_csv("train.csv.zip", compression='zip')

# Keep only relevant columns to match your sir's simplified workflow
cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 
        'Credit_Utilization_Ratio', 'Credit_Score']

# Clean data: drop missing values
df = df[cols].dropna()

print(df.head())
sns.countplot(x='Credit_Score', data=df)
plt.title("Distribution of Credit Scores")
plt.show()

sns.boxplot(x='Credit_Score', y='Outstanding_Debt', data=df)
plt.title("Debt vs Credit Score")
plt.show()
# Insight: Higher Outstanding Debt correlates with 'Poor' Credit Scores.
# Map Target: Poor=0, Standard=1, Good=2
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])
# Create the required ratios
df['debt_ratio'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
df['payment_delay_impact'] = df['Num_of_Delayed_Payment'] * df['Interest_Rate']
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Using XGBoost as it handles classification better than RF for this specific dataset
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance for Credit Score")
plt.show()
params = {'n_estimators': [100, 150], 'max_depth': [3, 6]}
grid = GridSearchCV(XGBClassifier(), params, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
pickle.dump(best_model, open("credit_model.pkl", "wb"))
print("Model Saved Successfully!")
