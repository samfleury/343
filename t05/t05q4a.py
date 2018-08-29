import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

x = diabetes.data
y = diabetes.target


num_points, num_attribute = x.shape
# Create a random permutation of indexes of all the points
I = np.random.permutation(num_points)
# Rearrange the images and labels according to the new random order
x = x[I, :]
y = y[I]

# Pick the split at somewhere near the 80% of indexes
split = int(np.floor(num_points * 0.8))
# For training take the indexes up to the split (80% of points)
xtrain = x[0:split, :]
ytrain = y[0:split]
# For testing take the indexes after the split (remaining 20%)
xtest = x[split:, :]
ytest = y[split:]

# Create a MLP regressor model
clf = MLPRegressor(alpha = 0.8, hidden_layer_sizes = (8, 16, 16, 16, 4), activation = 'relu', max_iter = 3000)

# Train the network
clf.fit(xtrain, ytrain)


yhat = clf.predict(x)
mse = np.mean(np.square(y-yhat))

print(mse)

# print(clf.score(xtrain, ytrain))
# print(clf.score(xtest, ytest))
