# import python package
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import read_csv

# read databaset
path = r"..\csv_data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, skiprows = 9, names = headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
print ('data.shape:\n', data.shape)    # (768, 9)
print ('data.head():\n', data.head())

# plot as hierarchical cluster.
patient_data = data.iloc[:, 3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (10, 7))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(data, method = 'ward'))
plt.show()