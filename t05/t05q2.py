import numpy as np
import os
import inspect
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

workingDir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
npzfile = np.load(os.path.join(workingDir, "t5dataset1.npz"))
x = npzfile['x']
y = npzfile['y']

# Create a MLP regressor model
clf = MLPRegressor(alpha = 0.08, hidden_layer_sizes = (50, 30, 10, 30,  10, 5), activation = 'relu', max_iter = 2000)

# Train the network
clf.fit(x, y)


yhat = clf.predict(x)
mse = np.mean(np.square(y-yhat))

print(mse)

plt.close('all')
plt.figure()

plt.plot(x, y, 'k.')

plt.plot(x, clf.predict(x), 'g-')

plt.show()