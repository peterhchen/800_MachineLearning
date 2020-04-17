from numpy import loadtxt
path = r"..\csv_data\pima-indians-diabetes.csv"
datapath = open (path, 'r')
data = loadtxt (datapath, delimiter=',')
print ('data.shape:', data.shape)
print('data[:3]', data[:3])