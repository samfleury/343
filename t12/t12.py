from gridworld import gridworld
import numpy as np
import random

env = gridworld()

Q = np.zeros((env.num_states, env.num_actions))
epochs = 10000
alpha = 0.05

s, R = env.reset()

for i in range(epochs):
    if np.random.random() < 1.0 - 0.9 * i/epochs:
        a = np.random.randint(0, env.num_actions)
        print(env.num_actions)
    else:
        a = np.argmax(Q[s,:])

    sprime, Rprime = env.step(a)

    Q[s, a] = Q[s, a] + alpha * (R + np.max(Q[sprime, :]) - Q[s, a])

    if env.terminal():
        Q[sprime, :] = Rprime
        s, R = env.reset()
    else:
        s = sprime
        R = Rprime
    env.show(Q=Q)
