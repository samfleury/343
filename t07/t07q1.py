import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Import the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Compute the mean value of each dimension in X
muX = np.mean(X, axis=0)
# Subtract mean value from X - always do this before PCA
X -= muX

# Create PCA model
h_pca = PCA()
# Compute the PCA components
h_pca.fit(X)

# Create the KMeans model
K = 3
h_kmeans = KMeans(n_clusters=K)
# Compute the centeroids
h_kmeans.fit(X)
# Get the labels of the datapoints based on the compute centeroids
y_kmeans = h_kmeans.predict(X)
# Get the centeroids
C = h_kmeans.cluster_centers_

Cpca = h_pca.transform(C)

print("Eigenvectors:")
print(h_pca.components_)
print("Eigenvalues:")
print(h_pca.explained_variance_ratio_)

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

