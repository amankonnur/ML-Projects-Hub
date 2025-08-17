import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = pd.read_csv("Iris.csv")

X = iris.drop("Species",axis=1)
y=iris['Species']

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# define the classifier
knn = KNeighborsClassifier(n_neighbors=3)

# train / fit the model
knn.fit(X_train,y_train)

# prediction.
y_prediction = knn.predict(X_test)

# print(y_prediction)
print("Accuracy Score : ")
print(accuracy_score(y_test,y_prediction))
print("Classification Report : ")
print(classification_report(y_test,y_prediction))
print("Confusion Matrix : ")
print(confusion_matrix(y_test,y_prediction))

