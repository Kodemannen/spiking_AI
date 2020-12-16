import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import seaborn as sns
sns.set()

from game.game import CarGame
from snn.snn import SNN


#----------------------------------------------------------------------------------------------
# Here I will control the snn instance and the game instance and feed information between them





def split_pixels(pixels, nx=100, ny=50):
    '''
    Divide the pixels into compartments that are used as input to single neurons
    '''

    img_length = pixels.shape[1]   # length is n columns
    img_height = pixels.shape[0]   # height is n rows

    splitted = []

    i = 0
    j = 0
    done = False
    while not done:

        cutout = pixels[j:j+ny, i:i+nx] 
        splitted.append(cutout) 

        #plt.imshow(pixels)
        #plt.pause(0.05)
        #pixels[j:j+ny, i:i+nx] = 1

        # Move to next cutout:
        i += nx
        if i >= img_length - 1:
            i = 0
            j += ny

        # Check if done:
        if j >= img_height:
            done = True

    #plt.show()
        
    return np.array(splitted)



def get_input_vector(pixels, nx, ny):

    splitted = split_pixels(pixels, nx=nx, ny=ny)
    input_vector = np.sum(splitted, axis=(1,2))

    return input_vector




def main():

    #----------------------
    # Get game:
    game = CarGame()

    #----------------------
    # Get spiking neural network:
    snn = SNN(snn_config=None, 
              n_excitatory=5, 
              n_inhibitory=4, 
              n_inputs=10, 
              n_outputs=10, 
              use_noise=False,
              dt=0.1,
              input_node_type='poisson_generator'
              #input_node_type='spike_generator'
              )

    fps = 30

    #----------------------
    # Start playing
    playing = True

    while playing:

        game.play_one_step()
        pixels = game.get_pixels()  # input for the snn
        pixels = pixels.T.astype(np.float)

        pixels[:,:] /= 10053375


        square_side_x = 100           # n pixels per square that represents a retinal neuron 
        square_size_y = 50

        #n_input_neurons = 4
        #input_vector = get_input_vector(pixels, nx=square_side_x, ny=square_size_y) 
        # input_vector.shape = (n_neurons, ) 

        splitted = split_pixels(pixels, nx=square_side_x, ny=square_size_y)
        print(splitted.shape) 

        neuron1 = pixels[0:50,400:500]
        neuron2 = pixels[50:100,400:500]

        neurons = [neuron1, neuron2]

        firing_rates = np.sum(neurons, axis=(1,2))

        #snn.simulate()


        #plt.imshow(pixels)
        #plt.savefig('output/testfig.png')



if __name__=='__main__':
    main()
