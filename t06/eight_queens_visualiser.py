import pylab as pl
import numpy as np
import time

__author__ = "Lech Szymanski"
__email__ = "lechszym@cs.otago.ac.nz"


class chess_board:
    
    def __init__(self):
        #Create a new figure
        self.plfig=pl.figure(dpi=100)
        #Create a new subplot
        self.plax = self.plfig.add_subplot(111)
        #Create bitmap for the chessboard
        b = np.matrix('1 0 1 0 1 0 1 0; 0 1 0 1 0 1 0 1;1 0 1 0 1 0 1 0; 0 1 0 1 0 1 0 1;1 0 1 0 1 0 1 0; 0 1 0 1 0 1 0 1;1 0 1 0 1 0 1 0; 0 1 0 1 0 1 0 1');
        #Plot the chessboard
        self.plax.matshow(b, cmap=pl.cm.gray)
        pl.ion()
        pl.show()        
        self.scatter_handle = []

    #Show state of the board (encoded as an array of 8 queens with position
    #from the bottom of the board in each column)
    def show_state(self,c):
        if self.scatter_handle:
            self.scatter_handle.remove()
        #The queens are shown as red dots on the chessboard
        self.scatter_handle = self.plax.scatter(x=[0,1,2,3,4,5,6,7],y=[8-i for i in c], s=40, c='r')
        self.plfig.canvas.draw()
        # Close any open figures, and start a new one
        pl.pause(0.01)
        time.sleep(0.01)


if __name__ == '__main__':
    #Create instance of chess board visualisation
    board = chess_board()
    #Show 5 different random queen distributions
    for i in range(0,5):
        #Generate a random queen distribution - 8 integers in range 1-8
        c=np.random.randint(1, 8, 8)
        #Show the new state
        board.show_state(c)
        time.sleep(0.5)
