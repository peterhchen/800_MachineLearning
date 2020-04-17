from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
dataframe = read_csv (path, skiprows=9, names = headernames)
print('dataframe[:10]:\n', dataframe[:10])
array = dataframe.values
std_scaler = StandardScaler().fit(array)
std_scalered = std_scaler.transform(array)
set_printoptions (precision=2)
print ("\nstd_scalered:\n", std_scalered[0:10])