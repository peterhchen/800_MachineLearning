# import Python Packages
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

# Generate 2D, containing 4 blobs (spots).
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs (n_samples = 400, centers = 4, cluster_std = 0.60, random_state = 0)

print ('y_true:\n', y_true)
print ('X[:10, 0]:\n', X[:10, 0])
print ('X[:10, 1]:\n', X[:10, 1])

plt.scatter (X[:, 0], X[:, 1], s = 20)
plt.show()



# plt.scatter (X[:, 0], X[:, 1], c = y_kmens, s = 20, cmp = 'summer')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], center[:, 1], c = 'blue', s = 100, alpha = 0.9)
# plt.show()