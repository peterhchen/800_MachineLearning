# import Python Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics

# read data
boston = datasets.load_boston(return_X_y = False)
# 'filename': 'C:\Users\14088\AppData\Local\Programs\Python\Python38-32\lib\site-packages
# \sklearn\datasets\data\boston_house_prices.csv'
#print ('boston:\n', boston)
# path = r"..\csv_data\boston_house_prices.csv"
# boston1 = pd.read_csv (path, skiprows=2)
# print("boston1:", boston1) 
# boston = boston1.values
X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 1)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print('Coefficients: \n', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()