import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

from game.game import JumpGame
from snn.snn import SNN


#----------------------------------------------------------------------------------------------
# Here I will control the snn instance and the game instance and feed information between them




def main():

    #----------------------
    # Get game:
    game = JumpGame()

    #----------------------
    # Get spiking neural network:
    snn = SNN()

    #----------------------
    # Start playing
    playing = True

    while playing:

        game.play_one_step()
        pixels = game.get_pixels()  # input for the snn

        # use just a cutout from the pixels as input, to simplify
        input_ = pixels

        fig, ax = plt.subplots()

        ax.imshow(pixels.T)
        plt.savefig('output/game_testfig.png')

        # Create spikes out of the pixels?
        # a non zero pixel is a spike

        #snn.


        exit('jau')






if __name__=='__main__':
    main()
