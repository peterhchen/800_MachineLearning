# import python package
import matplotlib.pyplot as plt
import numpy as np

# Plot data point
X = np.array(
   [[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85],[62,80],[73,60],[87,96]])
print ('X[0]:\n', X[0])        # [7 8]
print ('X[1]:\n', X[1])        # [12, 20]
print ('X[0][0]:\n', X[0][0])  # 7
print ('X[0][1]:\n', X[0][1])  # 8
print ('X[:, 0]:\n', X[:, 0])  # [7, 12, 17, ..., 87] 
print ('X[:, 1]:\n', X[:, 1])  # [8, 20, 19, ..., 96]

labels = range(1, 11)
plt.figure(figsize = (10, 7))
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X[:,0],X[:,1], label = 'True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
   plt.annotate(
      label,xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')
plt.show()