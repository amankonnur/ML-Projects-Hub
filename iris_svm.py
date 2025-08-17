import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

iris = pd.read_csv("Iris.csv")

X = iris.drop("Species",axis=1)
y = iris['Species']

classes = ["Iris-setosa","Iris-virginica","Iris-versicolor"]

# train,test,split the data for testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# model
model = svm.SVC()

# train / fit
model.fit(X_train,y_train)

# predictions
y_pred = model.predict(X_test)


print(model)
print([i for i in y_test])
print([i for i in y_pred])
print("Accuracy_Score",accuracy_score(y_test,y_pred)*100)