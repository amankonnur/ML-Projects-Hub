from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("customer_data.csv")

X = df[["Age", "Income", "SpendingScore"]]
Y = df["Membership"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,Y_train)

predictions = model.predict(X_test)
probabilites = model.predict_proba(X_test)

for i in Y_test:
    print(i,end=" ")
# print("Actual : ",Y_test)
print("Predictions : ",predictions)
print("Accuracy : ",accuracy_score(Y_test,predictions))
print("Probabilities : ",probabilites)