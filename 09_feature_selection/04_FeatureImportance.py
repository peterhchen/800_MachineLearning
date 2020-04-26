# Feature Importance
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
from numpy import set_printoptions
path = r'..\csv_data\pima-indians-diabetes.csv'
headernames = ['preg', 'glucos', 'pres', 'skin', 'insulin', 'mass', 'diabetes', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
array = dataframe.values
# separate array into input X and output Y
X = array[:, 0:8]   # 0 .. 7
Y = array[:, 8]     # 8: class

set_printoptions(precision=5)
model = ExtraTreesClassifier ()
model.fit (X, Y)
print ("Feature Importance: %s\n" % model.feature_importances_)
# Each attribute has score.
# Feature Importance: [0.10792 0.23288 0.10367 0.07967 0.07492 0.13822 0.11818 0.14454]