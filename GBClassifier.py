from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")

# seperate the input and output data
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"].map({"Iris-setosa" : 0, "Iris-versicolor":1,"Iris-virginica":2})


# split the data 80/20

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# initialze the model

gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=0,
    learning_rate=0.1,
    max_depth= 4,
)

gb.fit(X_train,y_train)

predictions = gb.predict(X_test)

np_y_test = np.array([i for i in y_test])

print("A")
print("Predictions :")
print(predictions)
print(np_y_test)
print("Accuracy Score :")
print(accuracy_score(y_test,predictions))
