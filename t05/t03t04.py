import numpy as np
import matplotlib.pyplot as plt
import time
# Import datasets, classifiers, and performance metrics
from sklearn import datasets
from sklearn import preprocessing
# The digits dataset
digits = datasets.load_digits()
x = digits.data
y = digits.target

y = np.expand_dims(y,axis=1)
enc = preprocessing.OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

num_points, num_attribute = x.shape
# Create a random permutation of indexes of all the points
I = np.random.permutation(num_points)
# Rearrange the images and labels according to the new random order
x = x[I, :]
y = y[I, :]

# Pick the split at somewhere near the 80% of indexes
split = int(np.floor(num_points * 0.8))
# For training take the indexes up to the split (80% of points)
xtrain = x[0:split, :]
ytrain = y[0:split, :]
# For testing take the indexes after the split (remaining 20%)
xtest = x[split:, :]
ytest = y[split:, :]

# Randomised weight matrix, 64 weights each for 10 perceptrons
W = np.random.normal(0, 1, (10, 64))
# Randomised bias vector, 1 each for 10 perceptrons
b = np.random.normal(0, 1, 10)
alpha = 0.02

plot_handles = []

display_counter = 0


def train():
    #  This will give a xtrain x 10 matrix of output of all 10 perceptrons
    y_hat = yhat_train()

    # for each input:  (this means we're doing online training)
    #   calculate error for each weight + bias for each perceptron

    e = ytrain - y_hat

    # for each image
    for i in range(len (xtrain)):

        #for each perceptron
        for p in range(10):
    #   apply changes to weights and biases (W and b)

            # for each w of each p:
            change_W = e[i, p] * xtrain[i]

            # for each p:
            change_b = -e[i, p]

            # for each w of each p:
            W[p] = W[p] + alpha * change_W

            # for each p:
            b[p] = b[p] + alpha * change_b


def update():
    train()
    # calculate and display % correct
    #   need a method to compare each value of y and y_hat
    y_hat = yhat_train()
    # compare y and y_hat, display % correct
    error = 0
    for n in range(len(ytrain)):
        if np.sum(np.abs(ytrain[n] - y_hat[n])) > 0:
            error += 1
    print(error / len(y_hat))
    # display first 16 images
    display()


def test():
    # generate y_hat matrix + hard-limit output
    y_hat = yhat_test()
    # compare y and y_hat, display % correct
    error = 0
    for n in range(len(ytest)):
        if np.sum(np.abs(ytest[n] - y_hat[n])) > 0:
            error += 1

    print(error / len(y_hat))
    # display first 16 images
    display()


def yhat_train():
    y_hat = np.matmul(xtrain, W.transpose()) - b
    y_hat[y_hat >= 0] = 1
    y_hat[y_hat <0] = 0
    return y_hat


def yhat_test():
    y_hat = np.matmul(xtest, W.transpose()) - b
    y_hat[y_hat >= 0] = 1
    y_hat[y_hat <0] = 0
    return y_hat


def data_plot(x, y=None):
    global plot_handles

    num_points, num_attributes = x.shape
    im_height = 8
    im_width = 8

    if not plot_handles:
        plt.close('all')
        figure_handle = plt.figure()
        plt.ion()
        plt.show()
        n = 0
        for r in range(4):
            for c in range(4):
                if n >= num_points:
                    continue
                im = x[n, :].reshape(im_height, im_width)
                n += 1
                ph = figure_handle.add_subplot(4, 4, n)
                plot_handles.append(ph)
                ph.imshow(im)
                ph.xaxis.set_visible(False)
                ph.yaxis.set_visible(False)

    for n in range(len(plot_handles)):
        if np.sum(y[n, :]) != 1:
            class_label = "?"
        else:
            class_label = np.argmax(y[n, :])

        plot_handles[n].set_title(class_label)

    plt.pause(0.1)
    time.sleep(0.1)


def display():
    global display_counter
    if display_counter == 4:
        data_plot(xtrain, yhat_train())
        display_counter = 0
    else:
        display_counter += 1


for i in xtrain[0]:
    update()

print("final % error is")
test()