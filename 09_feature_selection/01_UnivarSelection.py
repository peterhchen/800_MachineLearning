# Univariable Selection
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
path = r'..\csv_data\pima-indians-diabetes.csv'
headernames = ['preg', 'glucos', 'pres', 'skin', 'insulin', 'mass', 'diabetes', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
array = dataframe.values
# separate array into input X and output Y
X = array[:, 0:8]   # 0 .. 7
Y = array[:, 8]     # 8: class
# Select the best features form dataset
# select teh best fit feature = 2 out of 8 features
test = SelectKBest (score_func=chi2, k=2)   
fit = test.fit(X, Y)
# set precision = 2
set_printoptions(precision=1)
print("\nX.shape\n", X.shape)   # (768,8)
print("\nX:\n", X[0:5, 0:8])
print("\nY.shape:\n", Y.shape)  # (768,)
print("\nY[:10]:\n", Y[0:5])
print("\nfit.scores_\n", fit.scores_)  # R^2 score of each column
# transform: calc mean and fill missing data
featured_data = fit.transform (X)
print("\nFeatured data:\n", featured_data[0:5])  # display 2 features x 0..5