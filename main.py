import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

from game import JumpGame
from snn import SNN


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

        # Create spikes out of the pixels?
        # a non zero pixel is a spike

        #snn.








if __name__=='__main__':
    main()
