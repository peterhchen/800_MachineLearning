# import python package
import pandas as pd
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# read data
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'label']
data = pd.read_csv(path, skiprows= 9, names = headernames)
array = data.values
#print ('array[:5]:\n', array[:5])
X = data[headernames]     # Features
y = data.label            # Target variable

# Setup parameter
alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha = alphas)

# Setup grid search
model = Ridge()
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid.fit(X, y)

# print search
print('grid.best_score_:\n', grid.best_score_)
print('grid.best_estimator_.alpha:\n', grid.best_estimator_.alpha)