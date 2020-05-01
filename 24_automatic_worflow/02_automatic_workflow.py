# import python package
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# read data
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'label']
data = pd.read_csv(path, skiprows= 9, names = headernames)
array = data.values
#print ('array[:5]:\n', array[:5])
X = data[headernames]     # Features
y = data.label            # Target variable

# Create features
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# Create pipeline and model
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)

# Evaluate
kfold = KFold(n_splits = 20, shuffle = True, random_state = 7)
results = cross_val_score(model, X, y, cv = kfold)
print('result.mean():\n', results.mean())