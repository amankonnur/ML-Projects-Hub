from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from matplotlib import pyplot as plt

boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data, boston.target

# print(X.shape,y.shape)
plt.scatter(X["RM"], y, alpha=0.7)
# plt.scatter(X,y)
plt.show()



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
