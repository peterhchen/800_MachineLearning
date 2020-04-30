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

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
# print ('labels:\n', labels) # colors: [2, 0, 1, 2]
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']
# print ('colors:\n', colors)  # ['r.' (0), 'g.' (1), ...., 'm.' (69) ]
#print ('len(X):\n', len(X))   # len(X): 700

# labels[0-699] = 0, 1, 2 => 'r.', 'g.', 'b.'
for i in range(len(X)):
    #print ('i:', i, 'labels[i]:', labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
   marker = ".", color = 'k', s = 20, linewidths = 5, zorder = 10)
plt.show()