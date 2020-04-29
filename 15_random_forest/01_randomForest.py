# import Python Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read data
path = r"..\csv_data\iris.csv"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, skiprows=1, names = headernames)
# get the top 5 rows.
data_top = dataset.head()
print ('data_top:\n', data_top)
print ('dataset[:10]:\n', dataset[:10])

# Data Preprocessing
# iloc: integer indexer location.
# iloc[:, -1]: Last column of data, iloc[:, 4]: fourth column of all data.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# print ('dataset.iloc[:, :-1].values:\n', dataset.iloc[:, :-1].values)
# print ('dataset.iloc[:, 4].values:\n', dataset.iloc[:, 4].values)
print ('X[:10]:\n', X[:10])
print ('y[:10]:\n', y[:10])
# split the dataset into 70% training data and 30% of testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# RandomForestClassifier class of sklearn 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)
# predict
y_pred = classifier.predict(X_test)

# print the result
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)