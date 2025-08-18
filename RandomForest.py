# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,classification_report
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("Telco-Customer-Churn.csv")
#
# # Clean and encode Churn column
# df["Churn"] = df["Churn"].str.strip().str.lower()
# df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})
# df = df.dropna(subset=["Churn"])
# df.replace(" ", np.nan, inplace=True)
# df["Churn"] = df["Churn"].fillna(0).astype(int)
#
#
#
# # Fix TotalCharges
# df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
# df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
#
#
# # remove the blank spaces
# df["Churn"] = df["Churn"].fillna("No").map({"Yes": 1, "No": 0})
#
# # Binary mappings
# df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
# df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
# df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
# df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
# df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
# df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
# df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
# df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
# df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
# df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
# df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
#
# # Multi-category mappings
# df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
# df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
# df['PaymentMethod'] = df['PaymentMethod'].map({
#     'Electronic check': 0,
#     'Mailed check': 1,
#     'Bank transfer (automatic)': 2,
#     'Credit card (automatic)': 3
# })
#
# # Target column
# df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#
# # Binary columns
# df['PhoneService']   = df['PhoneService'].map({'Yes': 1, 'No': 0})
# df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
#
# # Multiple-category columns → use get_dummies (One-Hot Encoding)
# df = pd.get_dummies(df, columns=[
#     'MultipleLines',
#     'InternetService',
#     'OnlineSecurity',
#     'OnlineBackup',
#     'DeviceProtection',
#     'TechSupport',
#     'StreamingTV',
#     'StreamingMovies',
#     'Contract',
#     'PaymentMethod'
# ], drop_first=True)
#
# # Target variable (y)
# df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#
# # Train features and target
# X = df.drop(['Churn', 'customerID'], axis=1)
# y = df['Churn']
#
#
# # split
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
#
# # model
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=20,
#     random_state=42,
# )
#
# # train / fit
# rf.fit(X_train,y_train)
#
# # predict
#
# y_pred = rf.predict(X_test)
#
# yn = ["YES" if i == 1 else "NO" for i in y_pred]
#
# y = [1 for i in y_pred if i == 1]
# n = [0 for i in y_pred if i == 0]
#
# churn_yes = sum(y_pred == 1)
# churn_no  = sum(y_pred == 0)
#
# print("Predictions are : ",yn)
# print("Accuracy is : ",accuracy_score(y_test,y_pred))
# print("Classification Report")
# print(classification_report(y_test,y_pred))
# print(f"Among {len(y_pred)} the company is going to loose {n} customers and will retain {y} customers.")
#
# print(f"Among {len(y_pred)} customers, the company is going to lose {churn_yes} and retain {churn_no}.")
#
# print(df["Churn"].isna().sum())   # should print 0



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


df = pd.read_csv("Telco-Customer-Churn.csv")

# 12: Drop rows only where Churn is missing, not the whole df
df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()   # 14: Force string + clean spaces
df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})              # 15: Map yes/no to 1/0
df["Churn"] = df["Churn"].fillna(0).astype(int)                 # 16: Fill any unmapped as 0 (assume No)


# Fix TotalCharges
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# --- REMOVED duplicate Churn mapping block here (was conflicting) ---


# Binary mappings
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 2})

# Multi-category mappings
df['InternetService'] = df['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
})

# --- REMOVED extra "Target column" remap block (already handled above) ---

# Binary columns
df['PhoneService']   = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# Multiple-category columns → use get_dummies (One-Hot Encoding)
df = pd.get_dummies(df, columns=[
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
], drop_first=True)


# Train features and target
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']


# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
)

# train / fit
rf.fit(X_train, y_train)

# predict
y_pred = rf.predict(X_test)

yn = ["YES" if i == 1 else "NO" for i in y_pred]

y = [1 for i in y_pred if i == 1]
n = [0 for i in y_pred if i == 0]

churn_yes = sum(y_pred == 1)
churn_no  = sum(y_pred == 0)

print("Predictions are : ", yn)
print(y_pred)
print("Accuracy is : ", accuracy_score(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))
print(f"Among {len(y_pred)} the company is going to loose {len(y)} customers and will retain {len(n)} customers.")

print(f"Among {len(y_pred)} customers, the company is going to lose {churn_yes} and retain {churn_no}.")

print(df["Churn"].isna().sum())   # should print 0