from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data  # features
y = iris.data[:, 2]  # Let's predict petal length (column index 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient Boosting Regressor (linear regression style loss)
gbr = GradientBoostingRegressor(loss="squared_error", random_state=42)
gbr.fit(X_train, y_train)

# Predictions
y_pred = gbr.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
