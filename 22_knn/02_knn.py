# import python package
import numpy as np
import pandas as pd
# dataset path
path = r"../csv_data/iris.csv"
# Assign the column headers
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# read dataset
dataset = pd.read_csv(path, skiprows = 1, names = headernames)
print ('dataset.head():\n', dataset.head())
# data preprocessing
array = dataset.values
X = array[:,:2]
y = array[:,2]
print ('dataset.shape:\n', dataset.shape)    # output:(150, 5)

# 60% training, 40% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

# KNN Regresssor
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 10)
knnr.fit(X, y)

# Print MSE
print ("The MSE is:",format(np.power(y-knnr.predict(X),2).mean()))
