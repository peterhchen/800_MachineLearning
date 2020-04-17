from pandas import read_csv
path = r"..\csv_data\iris.csv"
data = read_csv (path)
print('data.shape:', data.shape)
print('data[:3]', data[:3])