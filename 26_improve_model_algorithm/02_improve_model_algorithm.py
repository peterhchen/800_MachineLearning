# import python package
import pandas as pd
import numpy
from pandas import read_csv
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# read data
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'label']
data = pd.read_csv(path, skiprows= 9, names = headernames)
array = data.values
#print ('array[:5]:\n', array[:5])
X = data[headernames]     # Features
y = data.label            # Target variable

# Setup parameter
param_grid = {'alpha': uniform()}
# Setup Ridge Regression algorithm
model = Ridge()
random_search = RandomizedSearchCV(
   estimator = model, param_distributions = param_grid, n_iter = 50, random_state=7)
random_search.fit(X, y)
# print result
print('random_search.best_score_:\n',random_search.best_score_)
print('random_search.best_estimator_.alpha:\n', random_search.best_estimator_.alpha)