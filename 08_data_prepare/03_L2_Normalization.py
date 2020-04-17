from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
print('dataframe[:10]:', dataframe[:10])
array = dataframe.values
# print('array.shape:', array.shape)
# print('array[:10]\n', array[:10])
Data_normalizer = Normalizer(norm='l2').fit(array)
Data_normalized = Data_normalizer.transform(array)
set_printoptions (precision=2)
print ("\nNormalized Data:\n", Data_normalized[0:10])