from eightpuzzle import eightpuzzle
import numpy as np
import time

puzzle = eightpuzzle(mode='hard')
init_state = puzzle.reset()


# puzzle.show(state=init_state)

class Node:
    def __init__(self, s, parent, cost, action):
        self.s = s
        self.parent = parent
        self.cost = cost
        self.action = action
        self.f = self.cost + get_h(s)


def get_h(state):

    cost = 0;
    goal_matrix = np.array(puzzle.goal()).reshape(3, 3)
    state_matrix = np.array(state).reshape(3, 3)

    for i in range(9):
        state_pos = np.where(state_matrix == i)
        goal_pos = np.where(goal_matrix == i)
        diff_x = np.sum(abs(goal_pos[0] - state_pos[0]))
        diff_y = np.sum(abs(goal_pos[1] - state_pos[1]))
        cost += diff_x
        cost += diff_y

    return cost

root = Node(s=init_state, parent=None, cost = 0, action = None)
open_list = list()
closed_list = list()

current_node = root
open_list.append(current_node)


start_time = time.time()

num_actions = 0

while not puzzle.isgoal(current_node.s):
    # for each child, create node + add to open list
    valid_actions = puzzle.actions(s=current_node.s)
    for action in valid_actions:
        new_state = puzzle.step(s=current_node.s, a=action)
        new_node = Node(new_state, current_node, current_node.cost + 1, action)
        # add to open list if not in closed list
        is_closed = False
        for i in closed_list:
            if i.s == new_node.s:
                is_closed = True
                break

        if not is_closed:
            open_list.append(new_node)
            closed_list.append(new_node)

        # add to closed list

    # remove current from list (because it's been expanded, and children have reference)
    open_list.remove(current_node)

    smallest_in_list = open_list[0]
    # set current to smallest f in list
    for i in open_list:
        if i.f < smallest_in_list.f:
            smallest_in_list = i

    current_node = smallest_in_list

    num_actions += 1
    print(num_actions)

print(current_node.cost)
elapsed_time = time.time() - start_time
print("Elapsed time: %.1f seconds" % elapsed_time)

action_list = list()
while True:
    action_list.insert(0, current_node.action)
    if current_node.parent is not None:
        current_node = current_node.parent
    else:
        break

print(action_list)
puzzle.show(a=action_list)