from pandas import read_csv
from pandas import set_option
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
data = read_csv (path, skiprows=9, names = headernames)
print('data.shape:', data.shape)
print('data.dtypes:')
print(data.dtypes)
set_option('display.width',100)
set_option('precision', 2)

print('\nStatistical data:')
print(data.describe())

count_class = data.groupby ('class').size()
print('\nClass Distribution:')
print(count_class)