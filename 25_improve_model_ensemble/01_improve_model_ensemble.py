# import python package
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# read data
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'label']
data = pd.read_csv(path, skiprows= 9, names = headernames)
array = data.values
#print ('array[:5]:\n', array[:5])
X = data[headernames]     # Features
y = data.label            # Target variable

# give the input for 10-fold cross validation
kfold = KFold(n_splits = 10, shuffle = True, random_state = 7)

# Create sub-models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# Create voting ensemble model by combing the predictions of 
# above created sub-models
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, y, cv = kfold)
print('\nresults.mean():\n', results.mean())