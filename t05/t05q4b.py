import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

bcancer = datasets.load_breast_cancer()
x = bcancer.data
y = bcancer.target

print(x.shape)
print(y.shape)

num_points, num_attribute = x.shape
# Create a random permutation of indexes of all the points
I = np.random.permutation(num_points)
# Rearrange the images and labels according to the new random order
x = x[I, :]
y = y[I]


# Pick the split at somewhere near the 80% of indexes
split = int(np.floor(569 * 0.8))
# For training take the indexes up to the split (80% of points)
xtrain = x[0:split, :]
ytrain = y[0:split]
# For testing take the indexes after the split (remaining 20%)
xtest = x[split:, :]
ytest = y[split:]

clf = MLPClassifier(alpha = 1e-5, hidden_layer_sizes = (64, 32, 12), activation = 'relu', max_iter = 500)

clf.fit(xtrain, ytrain)

score = clf.score(xtrain, ytrain)

print(score)

score = clf.score(xtest, ytest)

print(score)