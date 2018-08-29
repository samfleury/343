# Import the tictactoe class
from tictactoe import tictactoe
# Import the numpy library
import numpy as np


class Node:
    def __init__(self, s, value):
        self.s = s
        self.value = value


# Create a subclass of tictactoe with an implementation of the agent_move function
class mygame(tictactoe):

    # The agent_move function - the function has to tell the tictactoe class what move to make next in the game

    # Input - current state of the board

    # Returns - new states of the board with the proposed move
    def agent_move(self, state):
        return max_value(self, state, 6, True)


"""
        # Currently the agent makes a random move. The state of the board is a 3x3 matrix with values of 1's
        # denoting the agent's marks, -1's denoting the opponent's marks, and 0's denoting the empty fields
        # (where a new mark can be made)

        while True:
            # Select the location of the new mark at random (generated 2 random numbers between 0 and 3
            # (but not including 3)
            r, c = np.random.randint(0, 3, 2)
            new_states = self.max_possible_moves(state)
            print(len(new_states))
            # Check if the new location on the board is unmarked - if so, make a mark and return the new state
            # otherwise try a different location
            if state[r, c] == 0:
                state[r, c] = 1
                return state
"""


def max_value(self, state, n, return_s=False):
        print("MAX")
        if self.terminal(state) or n == 0:
            return self.evaluate(state)

        # list of possible moves, removing symmetries
        new_states = self.max_possible_moves(state)
        new_states = self.remove_symmetries(new_states)

        # create list of children to evaluate
        open_list = list()

        for i in range(len(new_states)):
            new_node = Node(new_states[i], min_value(self, new_states[i], n - 1))
            open_list.append(new_node)

        max_v = -np.inf
        for n in open_list:
            if max_v < n.value:
                max_v = n.value
                next_state = n.s


        """
        # create list of best children
        best_children = list()

        # find best children
        # compare to first open initially
        best_children.append(open_list[0])
        for child in open_list:
            for current_best in best_children:
                if child.value > current_best.value:
                    best_children = list()
                    best_children.append(child)
                elif child.value == current_best.value:
                    best_children.append(child)

        # return random best child
        next_state = best_children[np.random.randint(0, len(best_children))].state
        """
        if return_s:
            return next_state
        else:
            #return best_children[0].value
            return max_v


def min_value(self, state, n):
        print("MIN")
        if self.terminal(state) or n == 0:
            return self.evaluate(state)

        # list of possible moves, removing symmetries
        new_states = self.min_possible_moves(state)
        new_states = self.remove_symmetries(new_states)

        # create list of children to evaluate
        open_list = list()

        for i in range(len(new_states)):
            new_node = Node(new_states[i], max_value(self, new_states[i], n - 1))
            open_list.append(new_node)

        """
        # create list of best children
        best_children = list()

        # find best children
        # compare to first open initially
        best_children.append(open_list[0])
        for child in open_list:
            for current_best in best_children:
                if child.value < current_best.value:
                    best_children = list()
                    best_children.append(child)
                elif child.value == current_best.value:
                    best_children.append(child)

        # return worst value
        return best_children[0].value
        """
        max_v = np.inf
        for n in open_list:
            if max_v > n.value:
                max_v = n.value

        return max_v


# Play this indefinitely
while True:
    # Run a game where the human makes the first move
    mygame(opponent="x")
    # Run a game where the agent makes the first move
    mygame(opponent="o")