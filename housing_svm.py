from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd

housing = pd.read_csv("housing.csv")

print(housing.head())