import csv
import numpy as np
path =r"..\csv_data\iris.csv"
with open (path,'r') as f:
    reader = csv.reader(f, delimiter =',')
    headers = next (reader)
    data = list(reader)
    #data = np.array(data).astype (float)
    data = np.array(data)
print('headers:', headers)       # Firs line is headers
print('data.shape:', data.shape) # (140, 5): data has 150 row and 5 columns
# i = 1
# for x in data:
#     print('i: ', i, ' x: ', x)
#     i = i + 1
print('data[:1]:', data[:1]) # print array from index 0
print('data[:5]:', data[:5]) # print array index 0-4