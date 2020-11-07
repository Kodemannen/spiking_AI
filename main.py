import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import seaborn as sns
sns.set()

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
        pixels = pixels.T.astype(np.float)

        # use just a cutout from the pixels as input, to simplify
        #input_ = pixels

        #print(input_1.shape) 


        pixels[:,:] /= 10053375

        #pixels
        #pixels[0:52, 400:500,] = 1 #np.random.randn(50, 100)


        square_side_x = 100           # n pixels per square that represents a retinal neuron 
        square_size_y = 50
        nx = 10
        ny = 2

        splitted = split_pixels(pixels)
        print(splitted.shape)
        exit('o')

        #inputs = [np.arange(i,i+step)]

        ## two lanes for now
        ## one eye per lane
        #for el in pixels.reshape(-1):
        #    if el != 0:
        #        print(el)
        #        print('ballll')

        #exit('i')

        #
        #img = np.ones(shape=(50,50)) * 0.3
        ##img = np.random.randn(40, 40)

        #fig, ax = plt.subplots()
        #ax.imshow(pixels, )
        ##ax.imshow(img,) 

        #plt.savefig('output/game_testfig.png')
        #exit('jau')
        #print()

        # Create spikes out of the pixels?
        # a non zero pixel is a spike

        #snn.



def split_pixels(img):
    '''
    Divide the pixels into compartments that are used as input to single neurons
    '''

    img_length = img.shape[1]   # length is n columns
    img_height = img.shape[0]   # height is n rows

    nx = 100
    ny = 50

    splitted = []

    i = 0
    j = 0
    done = False
    while not done:

        cutout = img[j:j+ny, i:i+nx] 
        splitted.append(cutout) 

        # Move to next cutout:
        i += nx
        if i >= img_length - 1:
            i = 0
            j += ny

        # Check if done:
        if j >= img_height:
            done = True
        
    return np.array(splitted)





if __name__=='__main__':
    main()
