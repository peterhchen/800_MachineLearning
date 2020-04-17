from pandas import read_csv
path = r"..\csv_data\pima-indians-diabetes.csv"
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
headernames = ['preg', 'glucose', 'press', 'skin', 'insulin', 'mass', 'pedigree', 'age', 'class']
data = read_csv (path, skiprows=9, names = headernames)
print('data.shape:', data.shape)
print('data[:10]')
print(data[:10])