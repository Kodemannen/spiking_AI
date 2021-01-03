import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import seaborn as sns
sns.set()

from matplotlib import collections as mc

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



def create_grid_line_box(n_lanes, n_neurons_per_lane, win_size):

    '''
    Creates a list containing the lines of a grid

    -----------------------------------------------------------------
    Parameter               : Type          | Description 
    -----------------------------------------------------------------

        n_lanes             : integer       | number of car lanes
        n_neurons_per_lane  : integer       | neurons per car lane

    '''

    n_vertical_lines   = n_neurons_per_lane + 1
    n_horizontal_lines = n_lanes + 1

    spacex = win_size[0]/n_neurons_per_lane             # horizontal cell space 
    spacey = win_size[1]/n_lanes                        # vertical cell space

    startx = 0                                          # left vertical edge
    starty = 0                                          # top horizontal edge

    endx = startx + spacex*n_neurons_per_lane           # right vertical edge
    endy = starty + spacey*n_lanes                      # bottom horizontal edge 

    line_box = []

    # Generating vertical lines:
    for i in range(n_vertical_lines):

        x = spacex*i + startx

        line = [(x, starty), (x, endy)]
        line_box.append(line)

    # Generating horizontal lines:
    for i in range(n_horizontal_lines):

        y = spacey*i + starty

        line = [(startx, y), (endx, y)]
        line_box.append(line)

    return np.array(line_box)



def plot_grid(ax, line_box):

    lc = mc.LineCollection(line_box, linewidths=1, color='black') # choose color here
    ax.add_collection(lc) 

    # Adjusting 'zoom level' to grid:
    xs = line_box[:,0,0]
    ys = line_box[:,1,1]

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())

    #ax.set_xlim([-10, 1000])
    #ax.set_ylim([-10, 350])

    return




def main():


    #----------------------
    # Hyper parameters:
    n_lanes = 2
    n_neurons_per_lane = 5


    #----------------------
    # Get game:
    win_size = 800, 100                         # units of pixels

    game = CarGame(win_size)

    line_box = create_grid_line_box(n_lanes, n_neurons_per_lane, win_size)
        
    dpi = 150
    fig, ax = plt.subplots(figsize=np.array(win_size)/dpi)
    plot_grid(ax, line_box)

    plt.axis('off')
    plt.savefig('testfig.png')
    

    exit('jallu')

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
