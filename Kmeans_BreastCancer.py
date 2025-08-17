from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# initialize the dataset
df = load_breast_cancer()

# assign the features and target variables
X = scale(df.data)
y=df.target

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# define the model
# define clusters here we have 2 [benign, malignant]
model = KMeans(n_clusters=2,random_state=0)

# train / fit the model
model.fit(X_train)

# predict now
predictions = model.predict(X_test)

labels = model.labels_

print("Actual : ",[i for i in y_test])
print("Predictions : ",predictions)
print("Labels : ",labels)
print("Accuracy",accuracy_score(y_test,predictions))





