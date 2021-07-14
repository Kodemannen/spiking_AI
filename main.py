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
# This script controls the snn and game instances and feed information between them
#----------------------------------------------------------------------------------------------

#-------------------------------------------------
# Game hyper-parameters:
#-------------------------------------------------
N_LANES = 2
N_CELLS_PER_LANE = 8            # must be even      
INPUT_CELL_INDICES = [6, 14] # indices of the cells in the background grid
                                # that are used as input to the snn

#----------------------------------------
# snn hyper parameters:
#----------------------------------------
N_EXCITATORY = 13
N_INHIBITORY = 5
N_INPUTS     = len(INPUT_CELL_INDICES)
N_OUTPUTS    = 2


NEST_DATA_PATH = 'output/nest_data'
JSON_PATH  = 'snn/default.json'




assert N_INPUTS == len(INPUT_CELL_INDICES)


#-------------------------------------------------
# animation settings: 
#-------------------------------------------------
FPS = 30                        





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
    #spacex = win_size[0]/N_CELLS_PER_LANE             # horizontal cell space 
    #spacey = win_size[1]/N_LANES                        # vertical cell space


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






def main():
    

    #     ____       _               
    #    / ___|  ___| |_ _   _ _ __  
    #    \___ \ / _ \ __| | | | '_ \ 
    #     ___) |  __/ |_| |_| | |_) |
    #    |____/ \___|\__|\__,_| .__/ 
    #                         |_|    

    #-------------------------------------------------
    # Game settings:
    #-------------------------------------------------
    win_size = win_width, win_height = 800, 100         # pixels



    #-------------------------------------------------
    # simulation settings:
    #-------------------------------------------------
    dt = 0.1                        # time resolution
    sim_time = 100                  # ms

    iterations_before_sim = 5
    # take the average of the pixels then?
    # hmm
    # yes




    #-------------------------------------------------
    # Get sizes/spaces of the cells in the grid:
    #-------------------------------------------------
    spacex = int(win_width  / N_CELLS_PER_LANE)             # horizontal cell space
    spacey = int(win_height / N_LANES)                      # vertical cell space
    cell_size = spacex*spacey


    #-------------------------------------------------
    # create game instance:
    #-------------------------------------------------
    obstacle_width = spacex
    obstacle_height= spacey

    game = CarGame(win_size,
                   obstacle_size=(obstacle_width, obstacle_height),
                   n_lanes=N_LANES,
                   n_cells_per_lane=N_CELLS_PER_LANE,
                   )

    game.add_background_lines()
    game.add_fov_lines(chosen_cells=INPUT_CELL_INDICES)

    


    #-------------------------------------------------
    # Add background grid
    #-------------------------------------------------
    #game.add_background_lines(background_lines=line_box) 


    #-------------------------------------------------
    # Highlight the field-of-view of the snn
    #-------------------------------------------------
    # game.create_fov_lines(INPUT_CELL_INDICES,
    #                       spacex,
    #                       spacey) 



    dpi = 150
    # fig, ax = plt.subplots(figsize=np.array(win_size)/dpi)
    # plot_grid(ax, line_box)

    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig('testfig.png')
    #exit('jall')
    


    #-------------------------------------------------
    # Create spiking neural network instance:
    #-------------------------------------------------
    sim_index = 0

    snn = SNN(snn_config=None, 
            n_excitatory=N_EXCITATORY, 
            n_inhibitory=N_INHIBITORY, 
            n_inputs=N_INPUTS, 
            n_outputs=N_OUTPUTS, 
            use_noise=False,
            dt=dt,
            #input_node_type='poisson_generator'
            input_node_type='spike_generator',
            nest_data_path=NEST_DATA_PATH,
            json_path=JSON_PATH,
            sim_index=sim_index,
            )
    snn.set_positions()
    snn.get_conn_lines()


    #-------------------------------------------------
    # Stuff
    #-------------------------------------------------
                                    # unit:
    # set velocity so that it 
    # moves one neuron box 
    game.obstacle_vel = spacex/1    # pixels
    game.delay_ms     = 10          # ms
    frame_batch_size  = 4           # number of frames before we feed data to snn

    
    #-------------------------------------------------
    # make obstacles move on the neuron grid 
    #-------------------------------------------------

    game.obstacle_width = spacex
    game.obstacle_height = spacey

    obstacle_sum = spacex*spacey    # this should be approximately
                                    # equal to the sum of 1's in 
                                    # a grid square, i.e. what a
                                    # neuron sees when the square
                                    # is fully covered by the obstacle

    T = 0       # accumulated time
    counts = 0  # game/sim iterations


    #-------------------------------------------------
    # data boxes:
    #-------------------------------------------------
    sim_data_box = []



    #       ____                        _                   
    #      / ___| __ _ _ __ ___   ___  | | ___   ___  _ __  
    #     | |  _ / _` | '_ ` _ \ / _ \ | |/ _ \ / _ \| '_ \ 
    #     | |_| | (_| | | | | | |  __/ | | (_) | (_) | |_) |
    #      \____|\__,_|_| |_| |_|\___| |_|\___/ \___/| .__/ 
    #                                                |_|    

    game.playing = True
    while game.playing:


        pixels_sum = 0

        game.play_one_step()


        # Can sum the pixels over the frame batch 
        # then the pixels that appear in the most will have the highest 
        # value. Like a long exposure on a camera.


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
        # we should run the snn after a set
        # number of game iteration
        #---------------------------------------------
        
        ######  ####   ####    #### 
          ##   ##  ##  ##  #  ##  ## 
          ##   ##  ##  ##  #  ##  ## 
          ##   ##  ##  ##  #  ##  ## 
          ##    ####   ####    ####

        #---------------------------------------------
        # Here we must convert the pixels in any given
        # cell to spikes
        #---------------------------------------------

        max_val = cell_size
        input_spikes = []

        for i in INPUT_CELL_INDICES:

            # indexing to skip the edges
            inp_ratio = np.sum(splitted[i][1:-1,1:-1]) / max_val
            #inp_ratio = 0.3

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
        sim_data = snn.simulate(input_spikes=input_spikes,
                               sim_time=sim_time,
                               T=T)

        # e_spike_times = sim_data[0]     # (neuron[i], spiketimes) 
        # i_spike_times = sim_data[1] 
        # input_spike_times = sim_data[2]
        output_spike_times = np.array(sim_data[3])
        print(output_spike_times.shape) 

        #rate_output = sim_data[4][2]   # avgeraged across neurons


        #---------------------------------------------
        # Make decision based on activity: 
        #---------------------------------------------
        
        # must choose some firing rate threshold
        # or we could have a node for "do nothing"
        
        threshold = 40  # hz
        print(output_spike_times, "-"*20)

        T += sim_time
        counts += 1




    #---------------------------------------------
    # After game over:
    #---------------------------------------------

    # So problem now is that we have no input spikes!
    # Why the fuck?
    # Why then do we still get excitatory spikes?


    
    # problem seems to be that the files are overwritten each simulate()
    # not a problem yet, but maybe it could be.. not sure..
    # if this is the problem --> need to get it to append to the file instead


    all_spike_data = snn.read_spikes_from_file()

    e_spike_times, i_spike_times, input_spike_times, output_spike_times  = all_spike_data.values()
    #e_spike_times, i_spike_times, input_spike_times, output_spike_times, _  = sim_data


    
    #e_spike_times = np.array(e_spike_times)
    #i_spike_times = np.array(i_spike_times)
    #input_spike_times = np.array(input_spike_times)
    #output_spike_times = np.array(output_spike_times)


    snn.animate(e_spike_times,      # shape=(n_excitatory, spike_times) 
                i_spike_times,      
                input_spike_times, 
                output_spike_times)




if __name__=='__main__':
    main()
