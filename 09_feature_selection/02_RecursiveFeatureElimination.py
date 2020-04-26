# Recursive Feature Elimination
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
path = r'..\csv_data\pima-indians-diabetes.csv'
headernames = ['preg', 'glucos', 'pres', 'skin', 'insulin', 'mass', 'diabetes', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
array = dataframe.values
# separate array into input X and output Y
X = array[:, 0:8]   # 0 .. 7
Y = array[:, 8]     # 8: class
# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
# model = LogisticRegression() 
# uses te default solver = "lbfgs", which generate the warning message. 
# change solver to 'liblibear' to suppress the error message
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, 3)     # Pick up the best 3 attributes
fit = rfe.fit (X, Y)
print ("Number of Features: %d \n" % rfe.n_features_to_select)
print ("Selected Features: %s\n" % rfe.support_)
print ("Feature Ranking: %s\n" % rfe.ranking_)
