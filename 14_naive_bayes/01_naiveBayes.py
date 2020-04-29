# import Python Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Import sklearn.datasets. 
# Generate blobs by make_blobls() function. 
from sklearn.datasets import make_blobs
X, y = make_blobs (300, 2, centers = 2, random_state = 2, cluster_std = 1.5)
print ('X[:10]:\n', X[:10])
print ('y[:10]:\n', y[:10])
plt.scatter (X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')
# Use GaussianNB
from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit (X, y)

# Make prediction
rng = np.random.RandomState (0)
Xnew = [-6, -14] + [14, 18] * rng.rand (2000, 2)
ynew = model_GNB.predict (Xnew)

yprob = model_GNB.predict_proba (Xnew)
print ('yprob [-10:].round(3):\n', yprob [-10:].round(3))
# Plot new data to find boundaries
plt.scatter (X[:, 0], X[:, 1], c=y, s = 50, cmap = 'summer')
lim = plt.axis()
plt.scatter (Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = 'summer', alpha = 0.1)
plt.show()
