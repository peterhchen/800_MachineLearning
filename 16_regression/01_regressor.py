# import Python Packages
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd

# read data
path = r"..\csv_data\linear.txt"
input_data = np.loadtxt(path, delimiter=',')
print ('input_data:\n', input_data)
# X[:, :-1]: all in row, column from start:stop:-1, y[:, -1]: all rows, last column only.
X, y = input_data[:, :-1], input_data[:, -1]
print ('X:\n', X)
print ('y:\n', y)

# Training and test dataset
training_samples = int (0.6 * len(X))
print ('training_samples:\n', training_samples)
testing_samples = len(X) - training_samples
print ('testing_samples:\n', testing_samples)
X_train, y_train = X[:training_samples], y[:training_samples]
print ('X_train:\n', X_train)
print ('y_train:\n', y_train)
X_test, y_test = X[testing_samples:], y[testing_samples:] 
print ('X_test:\n', X_test)
print ('y_test:\n', y_test)

# Model evaluation and prediction
reg_linear = linear_model.LinearRegression()
reg_linear.fit (X_train, y_train)
y_test_pred = reg_linear.predict (X_test)

# Performance compulation
print("\nRegressor model performance:")
print("Mean absolute error (MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error (MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Plot and Visualization
plt.scatter (X_test, y_test, color = 'red')
plt.plot (X_test, y_test_pred, color='black', linewidth = 2)
plt.xticks (())
plt.yticks (())
plt.show()