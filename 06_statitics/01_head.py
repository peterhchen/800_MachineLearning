from pandas import read_csv
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
data = read_csv (path, skiprows=9, names = headernames)
print('data.shape:', data.shape)
print('data[:20]')
print(data[:20])