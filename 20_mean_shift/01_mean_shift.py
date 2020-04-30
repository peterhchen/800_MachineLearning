# import pyhton package
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets.samples_generator import make_blobs
# Generated data based on centers
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)
# Plot based on X[:, 0] and X[:, 1]
plt.scatter(X[:,0],X[:,1])
plt.show()