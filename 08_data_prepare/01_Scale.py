from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
print('dataframe[:10]:', dataframe[:10])
array = dataframe.values
print('array.shape:', array.shape)
print('array[:10]\n', array[:10])
data_scaler = preprocessing.MinMaxScaler (feature_range=(0, 1))
data_rescaled = data_scaler.fit_transform (array)
set_printoptions (precision=1)
print ("\nScaled Data:\n", data_rescaled[0:10])