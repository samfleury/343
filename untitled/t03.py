import numpy as np
import matplotlib.pyplot as plt

x = np.array ([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

y = np.array ([0,
               0,
               0,
               1])

w = np.random.normal(0, 1, 2)
w_change = np.zeros(w.shape[0])
b_change = np.zeros(w.shape[0])

b = 0
a = .2


length = y.shape[0]
shape = x.shape

e = np.zeros(length)


def hypothesis():
    yh = np.zeros(length)
    for i1 in range(length):
        result = 0
        for i2 in range(x.shape[1]):
            result += w[i2] * x[i1][i2]
        result -= b
        if result >= 0:
            yh[i2] = 1
        else:
            yh[i2] = 0
    return yh


def error_check():
    for i in range(length):
        e[i] = y[i] - h[i]
    return


def calc_for_each_w(w_num):
    sum = 0
    for i in range(length):
        sum += e[i] * x[i][w_num]
    sum /= length
    return sum


def calc_w_for_each_sample():
    for i in range(w.shape[0]):
        w_change[i] = calc_for_each_w(i)
    return


def calc_change_b():
    for i in range(length):
        b_change[i] = (-1) * e[i]
    return

def assign_w():
    for i1 in range(w.shape[0]):
        # calculate average for each w_change
        sum = 0
        for i2 in range(length):
            sum += w_change[i2][i1]
        # assign average of each w_change to each w
        w[i1] = w[i1] + (a * w_change[i2][i1])
    return


def assign_b():
    # calculate average for b
    sum = 0
    for i in range(length):
        sum += b_change[i]
    sum /= length
    # assign average to b
    global b
    b = sum


h = hypothesis()
error_check()
calc_w_for_each_sample()
calc_change_b()
assign_w()  # think my math is wrong here
assign_b()






plt.close('all')
plt.figure()

# Get indices of points labelled as 0.
I0 = np.where(y == 0)
plt.plot(x[I0, 0], x[I0, 1], 'r.')

# Get indices of points labelled as 1
I1 = np.where(y == 1)
plt.plot(x[I1, 0], x[I1, 1], 'b.')

plt.show()
