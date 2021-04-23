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





def split_pixels(pixels, spacex=100, spacey=50):
    '''

    Splits an input image into squares, given 
    some chosen square size spacex*spacey

    -----------------------------------------------------------------
    Input argument          : Type          | Description 
    -----------------------------------------------------------------

        pixels              : numpy array   | input image
        spacex              : integer       | horizontal cell size
        spacey              : integer       | vertical cell size

    '''
    #spacex = win_size[0]/n_cells_per_lane             # horizontal cell space 
    #spacey = win_size[1]/n_lanes                        # vertical cell space


    img_length = pixels.shape[1]            # integer 
    img_height = pixels.shape[0]            # integer


    splitted = []

    i = 0
    j = 0
    done = False
    while not done:

        cutout = np.array(pixels[j:j+spacey, i:i+spacex]) 
        splitted.append(cutout) 

        # Move to next cutout:
        i += spacex
        if i >= img_length - 1:
            i = 0
            j += spacey

        # Check if done:
        if j >= img_height:
            done = True
        
    splitted = np.array(splitted)

    return splitted




def get_input_vector(pixels, spacex, spacey):

    splitted = split_pixels(pixels, spacex=spacex, spacey=spacey)
    input_vector = np.sum(splitted, axis=(1,2))

    return input_vector



def create_grid_line_box(n_lanes, n_cells_per_lane, win_size):
    '''

    Creates a list containing the lines of a grid

    -----------------------------------------------------------------
    Input argument              : Type          | Description 
    -----------------------------------------------------------------

        n_lanes                 : integer       | number of car lanes
        n_cells_per_lane      : integer       | neurons per car lane

    '''

    n_vertical_lines   = n_cells_per_lane + 1
    n_horizontal_lines = n_lanes + 1

    spacex = win_size[0]/n_cells_per_lane             # horizontal cell space 
    spacey = win_size[1]/n_lanes                        # vertical cell space

    startx = 0                                          # left vertical edge
    starty = 0                                          # top horizontal edge

    endx = startx + spacex*n_cells_per_lane           # right vertical edge
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



def plot_grid(ax, line_box, auto_adjust=True):

    lc = mc.LineCollection(line_box, linewidths=1, color='black') # choose color here
    ax.add_collection(lc) 

    if auto_adjust:

        #-------------------------------------------------
        # Adjusting 'zoom level' to grid:
        #-------------------------------------------------

        xs = line_box[:,0,0]
        ys = line_box[:,1,1]

        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())

    #ax.set_xlim([-10, 1000])
    #ax.set_ylim([-10, 350])

    return



def split_and_get_grid_lines(pixels, ):
    pass
    
    


def main():
    

            #####  ##### ###### ##  ##  ####
            #      ##      ##   ##  ##  ##  #
            #####  ####    ##   ##  ##  ####
                #  ##      ##   ##  ##  ##
            #####  #####   ##   ######  ##


    #-------------------------------------------------
    # Game settings:
    #-------------------------------------------------
    win_size = win_width, win_height = 800, 100         # pixels



    #-------------------------------------------------
    # simulation settings:
    #-------------------------------------------------
    dt = 0.1                        # time resolution
    sim_time = 100                   # ms



    #-------------------------------------------------
    # animation settings: 
    #-------------------------------------------------
    fps = 30                        



    #-------------------------------------------------
    # hyper-parameters:
    #-------------------------------------------------
    n_lanes = 2
    n_cells_per_lane = 8          # must be even      



    #-------------------------------------------------
    # create background grid:
    #-------------------------------------------------

    spacex = int(win_width  / n_cells_per_lane)             # horizontal cell space
    spacey = int(win_height / n_lanes)                        # vertical cell space
    cell_size = spacex*spacey

    line_box = create_grid_line_box(n_lanes, 
                                    n_cells_per_lane, 
                                    win_size)


    #-------------------------------------------------
    # create game instance:
    #-------------------------------------------------
    obstacle_width = spacex
    obstacle_height= spacey

    game = CarGame(win_size,
                    obstacle_size=(obstacle_width, obstacle_height),
                    n_lanes=n_lanes)



    #-------------------------------------------------
    # Add background grid
    #-------------------------------------------------
    game.add_background_lines(background_lines=line_box) 



    dpi = 150
    fig, ax = plt.subplots(figsize=np.array(win_size)/dpi)
    plot_grid(ax, line_box)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('testfig.png')
    #exit('jall')
    


    #-------------------------------------------------
    # Create spiking neural network instance:
    #-------------------------------------------------
    snn = SNN(snn_config=None, 
            n_excitatory=5, 
            n_inhibitory=4, 
            n_inputs=2, 
            n_outputs=10, 
            use_noise=False,
            dt=dt,
            #input_node_type='poisson_generator'
            input_node_type='spike_generator'
            )
    snn.set_positions()
    snn.get_conn_lines()


    #-------------------------------------------------
    # Stuff
    #-------------------------------------------------
                                    # unit:
    # set velocity so that it 
    # moves one neuron box 
    game.obstacle_vel = spacex/1      # pixels
    game.delay_ms     = 10        # ms

    
    #-------------------------------------------------
    # make obstacles move on the neuron grid 
    #-------------------------------------------------

    game.obstacle_width = spacex
    game.obstacle_height = spacey

    obstacle_sum = spacex*spacey    # this should be approximately
                                    # equal to the sum of 1s in 
                                    # a grid square, i.e. what a
                                    # neuron sees when the square
                                    # is fully covered by the obstacle

    T = 0       # accumulated time


    #-------------------------------------------------
    # data boxes:
    #-------------------------------------------------
    out_data_box = []

    #-------------------------------------------------
    # Start game loop
    #-------------------------------------------------
    game.playing = True
    while game.playing:

        game.play_one_step()

        pixels = game.get_pixels()  # input for the snn
        #print(pixels.shape) 

        pixels = pixels.T.astype(np.float)
        pixels[:,:] /= 10053375     # normalized to contain values in (0, 1)


        #input_vector = get_input_vector(pixels, spacex=square_side_x, spacey=square_size_y) 
        # input_vector.shape = (n_neurons, ) 


        splitted = split_pixels(pixels, spacex, spacey)     # (n_cells, *cell_shape)
        print(np.sum(splitted[6]))
        print(np.sum(splitted[14]))



        #---------------------------------------------
        # Here we must convert the pixels in any given
        # cell to spikes
        #---------------------------------------------

        max_val = cell_size
        input_spikes = []

        input_indices = [6, 14]
        for i in input_indices:

            # indexing to skip the edges
            inp_ratio = np.sum(splitted[i][1:-1,1:-1]) / max_val

            #-----------------------------------------
            # mapping the number of pixels inside the 
            # input neurons to firing rates:
            #-----------------------------------------

            freq = inp_ratio       # input_ratio*10/s 
            period = 1/freq 
             
            spikes = np.arange(0.1 if T==0 else 0, sim_time, step=period)
            input_spikes.append(spikes)
            

        #---------------------------------------------
        # Run snn simulation:
        #---------------------------------------------
        sim_out = snn.simulate(input_spikes=input_spikes,
                               sim_time=sim_time,
                               T=T)

        T += sim_time
        #out_data_box.append(sim_out)


        #print(input_spikes) 
        #for i in range(len(input_spikes)):
        #    input_spikes[i]     # need to 

        #exit('asd')

        #exit('jalla')



        


        #for ting in splitted:
        #    print(ting)
        #    print(np.sum(ting))
        #    print(ting.shape)

        #    #exit('asd')

        #print('==================')


        #print(splitted.shape) 

        #neuron1 = pixels[0:50,400:500]
        #neuron2 = pixels[50:100,400:500]

        #neurons = [neuron1, neuron2]

        #firing_rates = np.sum(neurons, axis=(1,2))

        #snn.simulate()


        #plt.imshow(np.flip(pixels, axis=0))
        #plt.savefig('testfig.png')
        #exit('sd')
        print(T)

        if T >= 100000:
            playing = False
            game.quit_game()


    #---------------------------------------------
    # After game over:
    #---------------------------------------------
    spike_data = snn.get_spikes(T=T, sim_time=sim_time)
    np.save('spike_data.npy', spike_data)

    e_spike_times, i_spike_times, input_spike_times, output_spike_times, _ = spike_data
    
    e_spike_times = np.array(e_spike_times)
    i_spike_times = np.array(i_spike_times)
    input_spike_times = np.array(input_spike_times)
    output_spike_times = np.array(output_spike_times)

    snn.animate(e_spike_times,      # shape=(n_excitatory, spike_times) 
                i_spike_times,      
                input_spike_times, 
                output_spike_times)




if __name__=='__main__':
    main()
