# import python package
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# read data
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'label']
data = pd.read_csv(path, skiprows= 9, names = headernames)
array = data.values
print ('array[:5]:\n', array[:5])
X = data[headernames]     # Features
y = data.label            # Target variable

# create pipeline and Model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

# Evaluate
kfold = KFold(n_splits = 20, shuffle = True, random_state = 7)
results = cross_val_score(model, X, y, cv = kfold)
print('result.mean():\n', results.mean())