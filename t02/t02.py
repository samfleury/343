# These commands import some of the libraries we'll be using.
import numpy as np
import matplotlib.pyplot as plt
import os, inspect

workingDir = os.path.dirname(
                os.path.abspath(
                    inspect.getfile(inspect.currentframe())
                )
            )
npzfile = np.load(os.path.join(workingDir, "t1dataset1.npz"))

x = npzfile['x']
y = npzfile['y']

Nx = x.shape[0]
Ny = y.shape[0]

w0 = -2.1
w1 = 0.25
w2 = 0.22

hx = np.linspace(-5, 5, Nx)
hy = w2*hx**2 + w1*hx + w0

# Close any open figures, and start a new one
plt.close('all')
# Create a new (empty) figure
plt.figure()

# Plot the function (in greeen)
plt.plot(x, y, 'k.')
plt.plot(hx, hy, 'g--')

# Add a legend and axis labels
plt.legend(['given points', 'my function'])
plt.xlabel('x')
plt.ylabel('y')

# Display the figure
plt.show()
