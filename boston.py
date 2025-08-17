from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data, boston.target

# convert to numeric numpy arrays
X = X.astype(float).values
y = y.astype(float).values

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#algorithm
l_reg = linear_model.LinearRegression()


# # print(X.shape,y.shape)
# plt.scatter(X["RM"], y, alpha=0.7)
# # plt.scatter(X,y)
# plt.show()


# train then model by fitting the data
model = l_reg.fit(X_train,y_train)

# predict the data
predictions = model.predict(X_test)

print([i for i in y_test])
print([p for p in predictions])

# evaluation
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))


# print("Actual values:", list(y_test[:10]))
# print("Predicted values:", list(predictions[:10]))

"""
CRIM → per capita crime rate by town
ZN → proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS → proportion of non-retail business acres per town
CHAS → Charles River dummy variable (1 = tract bounds river; 0 otherwise)
NOX → nitric oxides concentration (parts per 10 million)
RM → average number of rooms per dwelling
AGE → proportion of owner-occupied units built before 1940
DIS → weighted distances to five Boston employment centers
RAD → index of accessibility to radial highways
TAX → full-value property-tax rate per $10,000
PTRATIO → pupil-teacher ratio by town
B → 1000(Bk - 0.63)^2 where Bk is proportion of Black residents by town
LSTAT → % lower status of the population
"""
