from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("customer_data.csv")

# Features & Target
X = df[["Age", "Income", "SpendingScore"]]
Y = df["Membership"]

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # limit depth for clarity
model.fit(X_train, Y_train)

# Predictions
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, predictions))

# Plot the tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=["Age","Income","SpendingScore"],
          class_names=["Non-Member","Member"], filled=True)
plt.show()
