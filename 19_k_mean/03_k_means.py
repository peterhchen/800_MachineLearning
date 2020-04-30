# import Python Packages
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

# Load digits data
from sklearn.datasets import load_digits
digits = load_digits()
print ('digits.data.shape:\n', digits.data.shape)
print ('digits.data[:2, :64]:\n', digits.data[:2, :64])
# output: (1797, 64)

# Generate the 10 clusters based one 10 digits data
kmeans = KMeans(n_clusters = 10, random_state = 0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
print ('kmeans.cluster_centers_.shape:\n', kmeans.cluster_centers_.shape)
print ('kmeans.cluster_centers_[:2, :10]:\n', kmeans.cluster_centers_[:2, :10])
# output (10, 64)

# Display clusters of 10 digits
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# loop two array with zip
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
plt.show()

# Make cluster label
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]

# Compute accuracy
from sklearn.metrics import accuracy_score
print ('accuracy_score(digits.target, labels):\n', accuracy_score(digits.target, labels))