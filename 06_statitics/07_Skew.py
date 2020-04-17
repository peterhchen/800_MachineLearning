from pandas import read_csv
from pandas import set_option
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
data = read_csv (path, skiprows=9, names = headernames)
print('data.shape:', data.shape)

set_option('display.width',100)
set_option('precision', 2)
skews = data.skew ()
print('\nSkew:')
print(skews)