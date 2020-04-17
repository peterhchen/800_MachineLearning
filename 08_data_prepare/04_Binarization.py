from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
print('dataframe[:10]:', dataframe[:10])
array = dataframe.values
binarizer = Binarizer(threshold=0.5).fit(array)
Data_binarized = binarizer.transform(array)
set_printoptions (precision=1)
print ("\nData_binarized:\n", Data_binarized[0:10])