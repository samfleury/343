from datasets import datasets
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import MNIST data
mnist = datasets.load_mnist()
X = mnist.data
y = mnist.target

# Compute the mean value of each dimension in X
muX = np.mean(X, axis=0)
# Subtract mean value from X - always do this before PCA
X -= muX

# Create PCA model
h_pca = PCA()
# Compute the PCA components
h_pca.fit(X)

# Create the KMeans model
K = 10
h_kmeans = KMeans(n_clusters=K, n_init=1)
# Compute the centeroids
h_kmeans.fit(X)
# Get the labels of the datapoints based on the compute centeroids
y_kmeans = h_kmeans.predict(X)
# Get the centeroids
C = h_kmeans.cluster_centers_

Cpca = h_pca.transform(C)

Xpca = h_pca.transform(X)

fh = plt.figure(figsize=(8, 6))
fh.add_subplot(2, 1, 1)
plt.scatter(Xpca[:, 0], Xpca[:, 1], c=y, cmap=plt.cm.Set1)
plt.title("2D visualisation of iris data (true clusters)")

fh.add_subplot(2, 1, 2)
plt.scatter(Xpca[:, 0], Xpca[:, 1], c=y_kmeans, cmap=plt.cm.Set1)
plt.scatter(Cpca[:,0], Cpca[:,1], c=range(K),
             cmap=plt.cm.Set1,
             marker='s', edgecolor='black')

plt.show()

# Show pca components
mnist.show(h_pca.components_)
# Show cluster centers
mnist.show(C)

