import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# Load CSV
df = pd.read_csv("credit_card_fraud_data.csv")

# 1️⃣ Replace empty strings with NaN
df.replace(" ", np.nan, inplace=True)

# 2️⃣ Remove duplicate rows
df.drop_duplicates(inplace=True)

# 3️⃣ Handle missing values
# Example: Drop rows with NaN in important columns
df.dropna(subset=["amt", "is_fraud"], inplace=True)

# Or fill NaN with appropriate values
df["city"].fillna("Unknown", inplace=True)
df["state"].fillna("Unknown", inplace=True)
df["job"].fillna("Unknown", inplace=True)

# 4️⃣ Convert date column to datetime
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

# 5️⃣ Convert dob (date of birth) to datetime
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

# 6️⃣ Create new feature: age
df["age"] = (pd.Timestamp.now() - df["dob"]).dt.days // 365

# 7️⃣ Remove outliers in transaction amount
Q1 = df["amt"].quantile(0.25)
Q3 = df["amt"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["amt"] >= Q1 - 1.5*IQR) & (df["amt"] <= Q3 + 1.5*IQR)]

# 8️⃣ Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# 2. Preprocessing: features & target
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Train-test split (preserve class ratio with stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define & train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric='auc'
)
model.fit(X_train, y_train)

# 5. Predictions & evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))





