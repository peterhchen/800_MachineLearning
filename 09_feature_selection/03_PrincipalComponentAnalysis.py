# Principal Component Analysis (PCA)
from pandas import read_csv
from sklearn.decomposition import PCA
from numpy import set_printoptions
path = r'..\csv_data\pima-indians-diabetes.csv'
headernames = ['preg', 'glucos', 'pres', 'skin', 'insulin', 'mass', 'diabetes', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
array = dataframe.values
# separate array into input X and output Y
X = array[:, 0:8]   # 0 .. 7
Y = array[:, 8]     # 8: class

set_printoptions(precision=1)
pca = PCA (n_components = 3)
fit = pca.fit (X, Y)
print ("Explained Variance: %s\n" % fit.explained_variance_ratio_)
print ("fit.component_: %s\n" % fit.components_)
