import multiprocessing as mp
from joblib import delayed, Parallel
import sys

import numpy as np
np.random.seed(13)

import nest
nest.set_verbosity("M_WARNING")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc
import seaborn as sns
sns.set()

import json

from functions import *


def __get_spike_times_by_id(idx, times, senders, scale):
    m = (senders == idx)
    return times[m] / scale


def get_spike_times_by_id(times_senders, pop, unit_s=False):
    """
    Separate the spike times per neuron.

    :param times_senders: list of length 2 containing times and senders
    :param pop: list of nest neuron ids
    :param unit_s: return spike times in s instead of in ms, default: False
    :returns: list containing a list of spike times for each neuron
    """

    times, senders = times_senders

    if len(senders) == 0:
        return [], []

    # spikes_per_id = []
    scale = 1000. if unit_s else 1.

    delayed_ = (delayed(__get_spike_times_by_id)(idx, 
                                                 times, 
                                                 senders, 
                                                 scale) for idx in pop)
    ret_ = Parallel(n_jobs=mp.cpu_count())(delayed_)

    return ret_


class SNN:
    """
    Builds, simulates and visualizes a spiking neural network 
    """

    def __init__(self, 
                 snn_config=None, 
                 n_excitatory=5, 
                 n_inhibitory=4, 
                 n_inputs=10, 
                 n_outputs=10, 
                 use_noise=False,
                 dt=0.1
                 ):
        """
        Main pop : consists of excitatory and inhibitory synapses
        Input synapses: only excitatory synapses
        Output pop : no synapses 
        """

        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = dt
        self.use_noise = use_noise

        nest.ResetKernel()
        nest.SetKernelStatus(dict(
                            resolution=dt,
                            ))

        
        #-----------------------------------------------
        # Loading parameters:
        with open('default.json', 'r') as f:
          default_cfg = json.load(f)

        self.snn_conf = default_cfg.copy()
        if snn_config:
            self.snn_conf.update(snn_config)

        #-----------------------------------------------
        # Creating nodes:
        e_lif_params = self.snn_conf["e_lif_params"]
        i_lif_params = self.snn_conf["i_lif_params"]

        self.e_population = nest.Create('iaf_psc_alpha', 
                                        n_excitatory, 
                                        e_lif_params)

        self.i_population = nest.Create('iaf_psc_alpha', 
                                        n_inhibitory, 
                                        i_lif_params)

        self.input_nodes = nest.Create('spike_generator', 
                                        n_inputs)

        self.output_nodes = nest.Create('iaf_psc_alpha', 
                                        n_outputs, 
                                        e_lif_params)

        assert (self.e_population[0] == 1), '''Nodes must be created 
                                            first and has to be in the order 
                                            exc, inh, input, output'''

        #-----------------------------------------------
        # Creating spike detectors:
        self.e_spike_detector = nest.Create('spike_detector')
        self.i_spike_detector = nest.Create('spike_detector')
        self.input_spike_detector = nest.Create('spike_detector')
        self.output_spike_detector = nest.Create('spike_detector')

        #-----------------------------------------------
        # Connection rules and parameters:
        weight_e = 10.
        weight_i = -2.5
        weight_inp = 10.
        outdegree = 2
        delay = 1.
        
        syn_spec_ee = self.snn_conf['syn_spec_ee']
        syn_spec_ei = self.snn_conf['syn_spec_ei']
        syn_spec_ie = self.snn_conf['syn_spec_ie']
        syn_spec_ii = self.snn_conf['syn_spec_ii']
        syn_spec_inp = self.snn_conf['syn_spec_inp']


        rule_dict_e = self.snn_conf['rule_dict_e']
        rule_dict_i = self.snn_conf['rule_dict_i']
        rule_dict_inp = self.snn_conf['rule_dict_inp']
        rule_dict_output = self.snn_conf['rule_dict_output']

        #-----------------------------------------------
        # Connecting nodes:
        nest.Connect(self.e_population, 
                     self.e_population, 
                     rule_dict_e, 
                     syn_spec_ee)

        nest.Connect(self.e_population, 
                     self.i_population, 
                     rule_dict_e, 
                     syn_spec_ei)

        nest.Connect(self.i_population, 
                     self.i_population, 
                     rule_dict_i, 
                     syn_spec_ii)

        nest.Connect(self.i_population, 
                     self.e_population, 
                     rule_dict_i, 
                     syn_spec_ie)

        nest.Connect(self.input_nodes, 
                     self.e_population, 
                     rule_dict_inp, 
                     syn_spec_inp)

        nest.Connect(self.e_population, 
                     self.output_nodes, 
                     rule_dict_output, 
                     syn_spec_ee)

        #-----------------------------------------------
        # Connecting to spike detectors:
        nest.Connect(self.e_population, self.e_spike_detector)
        nest.Connect(self.i_population, self.i_spike_detector)
        nest.Connect(self.input_nodes, self.input_spike_detector)
        nest.Connect(self.output_nodes, self.output_spike_detector)

        return None


    def plot_connectome(self, 
                        ax,

                        radius_e=0.8, 
                        radius_i=0.5, 
                        
                        center_e=(0,0), 
                        center_i=(0,2), 

                        input_column_size=1, 
                        input_column_center=(-1.5,0.9),

                        output_column_size=1, 
                        output_column_center=(1.5,0.9),
                        ):
        '''
        Plots all the nodes and connections in the SNN
        '''

        np.random.seed(0)
        #-----------------------------------------------
        # Random generating main pop positions:
        positions_e = get_circular_positions(
                                            n_neurons=self.n_excitatory, 
                                            radius=radius_e, 
                                            center=center_e)

        positions_i = get_circular_positions(
                                            n_neurons=self.n_inhibitory, 
                                            radius=radius_i, 
                                            center=center_i)

            # The relationship between neuron GID and position is: 

                # For excitatory:
                    # position index = GID - 1

                # For inhibitory:
                    # position index = GID - 1 - self.n_excitatory

        # Generating input column:
        positions_input = get_vertical_line_positions(
                                            n_neurons=self.n_inputs,
                                            column_size=input_column_size,
                                            column_center=input_column_center)

        # Generating output column:
        positions_output = get_vertical_line_positions(
                                            n_neurons=self.n_outputs,
                                            column_size=output_column_size,
                                            column_center=output_column_center)

        #connections_.. = nest.GetConnections(target=target, source=source)   
        # format: (source-gid, target-gid, target-thread, synapse-id, port)
        # For an individual synapse we get:
            # source = connections_..[i][0]
            # target = connections_..[i][1]


        nest.GetConnections(self.e_population)


        #-----------------------------------------------
        # Connection storage:

        conn_groups = dict(
            conn_ee = nest.GetConnections(source=self.e_population, 
                                          target=self.e_population),
            conn_ei = nest.GetConnections(source=self.e_population, 
                                          target=self.i_population),
            conn_ii = nest.GetConnections(source=self.i_population, 
                                          target=self.i_population),
            conn_ie = nest.GetConnections(source=self.i_population, 
                                          target=self.e_population),
            conn_inp = nest.GetConnections(source=self.input_nodes, 
                                          target=self.e_population),
            conn_output = nest.GetConnections(source=self.e_population, 
                                          target=self.output_nodes),
            )

        conn_pairs = dict()
        for key in conn_groups:
            conn_pairs[key] = get_conn_pairs(conn_groups[key])   # shape (n_connections,2) 
        # conn_pairs[key] is the list with all connections from pop i to pop j
        # and each conn_pairs[key][k] is a tuple containing sender and receiver  


        #-----------------------------------------------
        #-------------------Plotting--------------------


        # Plotting connection lines:
        conn_lines = dict()
        for key in conn_pairs:

            if key=='conn_inp':
                sender_type = 'input'
                receiver_type = 'e'

            elif key=='conn_output':
                sender_type = 'e'
                receiver_type = 'output'

            else:
                sender_type = key[-2]
                receiver_type = key[-1]
            
            pairs = conn_pairs[key]
            n_pairs = len(pairs)
            lines = np.zeros(shape=(n_pairs, 2, 2))
            
            for i in range(n_pairs):
                pair = pairs[i]

                #--------------------------------------
                # Getting sender:
                if sender_type=='e':
                    source_pos_index = pair[0] - self.e_population[0] 
                    source_pos = positions_e[source_pos_index]

                elif sender_type=='i':
                    source_pos_index = pair[0] - self.i_population[0]
                    source_pos = positions_i[source_pos_index]

                elif sender_type=='input': 
                    source_pos_index = pair[0] - self.input_nodes[0]
                    source_pos = positions_input[source_pos_index]

                #--------------------------------------
                # Getting receiver:
                if receiver_type=='e':
                    receiver_pos_index = pair[1] - self.e_population[0]
                    receiver_pos = positions_e[receiver_pos_index]

                elif receiver_type=='i':
                    receiver_pos_index = pair[1] - self.i_population[0]
                    receiver_pos = positions_i[receiver_pos_index]

                elif receiver_type=='output':
                    receiver_pos_index = pair[1] - self.output_nodes[0] 
                    receiver_pos = positions_output[receiver_pos_index]

                lines[i, 0] = source_pos 
                lines[i, 1] = receiver_pos


            conn_lines[key] = lines
            lc = mc.LineCollection(lines, linewidths=0.1) # choose color here
            ax.add_collection(lc) 
            #_______________________________________________


        self.positions_e = positions_e
        self.positions_i = positions_i
        self.positions_input = positions_input
        self.positions_output = positions_output

        self.positions = dict(excitatory = positions_e,
                              inhibitory = positions_i,
                              input = positions_input,
                              output = positions_output)

        self.conn_pairs = conn_pairs
        self.conn_lines = conn_lines
            

        #-----------------------------------------------
        # Plotting nodes:
        
        ax.scatter(positions_e[:,0], 
                   positions_e[:,1], 
                   label='Excitatory')

        ax.scatter(positions_i[:,0], 
                   positions_i[:,1], 
                   label='Inhibitory')

        ax.scatter(positions_input[:,0], 
                   positions_input[:,1], 
                   color='grey', 
                   label='Input')

        ax.scatter(positions_output[:,0], 
                   positions_output[:,1], 
                   color='grey', 
                   label='Output')

        #plt.axis('off')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.legend()

        #self.base_ax = ax
        #self.base_fig = fig
        #plt.savefig('test.png')
        self.ax = ax

        return self.ax
    

    def __run_simulation(self, sim_time=100, T=0):

        #----------------------------------------------
        # simulate
        nest.Simulate(sim_time)

        #----------------------------------------------
        # analysis

        stat_e = nest.GetStatus(self.e_spike_detector, 'events')[0]
        stat_i = nest.GetStatus(self.i_spike_detector, 'events')[0]     

        stat_input = nest.GetStatus(self.input_spike_detector, 'events')[0]
        stat_output = nest.GetStatus(self.output_spike_detector, 'events')[0]

        # stat_x['times'] is a one dimensional list of spike times
        # stat_x['senders'] is a one dimensional list of gids 
        # corresponding to the spike times
        

        #----------------------------------------------
        # separating out the firings from the most recent simulation 
        # (after time T)

        times_e_indices  = np.argwhere( stat_e['times'] > T )[:,0]          
        times_i_indices  = np.argwhere( stat_i['times'] > T )[:,0]         
        times_input_indices = np.argwhere( stat_input['times'] > T )[:,0] 
        times_output_indices = np.argwhere( stat_output['times'] > T )[:,0]
        
        times_e = stat_e['times'][times_e_indices]
        times_i = stat_i['times'][times_i_indices]
        times_input = stat_input['times'][times_input_indices]
        times_output = stat_output['times'][times_output_indices]

        senders_e  = stat_e['senders'][times_e_indices]
        senders_i  = stat_i['senders'][times_i_indices]
        senders_input = stat_input['senders'][times_input_indices]
        senders_output = stat_output['senders'][times_output_indices]

        spikes_e = times_e, senders_e
        spikes_i = times_i, senders_i
        spikes_input = times_input, senders_input
        spikes_output = times_output, senders_output

        # extract spike times in an array per neuron
        e_spike_times = get_spike_times_by_id(spikes_e, self.e_population)
        i_spike_times = get_spike_times_by_id(spikes_i, self.i_population)

        input_spike_times = get_spike_times_by_id(spikes_input, 
                                                  self.input_nodes)

        output_spike_times = get_spike_times_by_id(spikes_output, 
                                                   self.output_nodes)


        if len(e_spike_times) == 0:
            print('no spikes')
            return

        # compute mean firing rates
        rate_e = len(times_e) * 1000.0 / (
                sim_time * float(self.n_excitatory))
        rate_i = len(times_i) * 1000.0 / (
                sim_time * float(self.n_inhibitory))
        rate_output = len(times_i) * 1000.0 / (
                sim_time * float(self.n_outputs))

        print('mean excitatory rate: {0:.2f} Hz'.format(rate_e))
        print('mean inhibitory rate: {0:.2f} Hz'.format(rate_i))

        return (e_spike_times, i_spike_times, input_spike_times, 
               output_spike_times, (rate_e, rate_i, rate_output))


    def simulate(self, input_spikes, sim_time, T=0):

        self.sim_time = sim_time
        input_spike_times = [{'spike_times': np.round(spt, 1) + T } 
                                            for spt in input_spikes]

        nest.SetStatus(self.input_nodes, input_spike_times)

        return self.__run_simulation(sim_time, T=T)

    
    def generate_spike_frames(self, 
                              e_spike_times,      # shape=(n_excitatory, spike_times) 
                              i_spike_times,      
                              input_spike_times, 
                              output_spike_times,
                              fps=30): 
        dt_anim = 1/fps * 1000          # in ms
        # should use the // operator to make a grid thing
        timesteps_anim = np.arange(0, self.sim_time, step=dt_anim)


        N = len(timesteps_anim)

        self.frames = dict(
            excitatory = [np.zeros(shape=(self.n_excitatory, N), dtype=np.int), e_spike_times],
            inhibitory = [np.zeros(shape=(self.n_inhibitory, N), dtype=np.int), i_spike_times],
            input = [np.zeros(shape=(self.n_inputs, N), dtype=np.int), input_spike_times],
            output = [np.zeros(shape=(self.n_outputs, N), dtype=np.int), output_spike_times]
            )

        for key in self.frames: 
            frame_matrix = self.frames[key][0]
            spike_times = self.frames[key][1]   # shape (n_nodes, x)


            #----------------------------------------------
            # Rounding up to grid:
            rounded = (spike_times // dt_anim) * dt_anim   # rounded to the left
            #rest = spike_times - rounded
            #to_add = (rest > (dt_anim/2)) * dt_anim     
            #rounded = rounded + to_add                     # rounded to closest 


            #-------------------------------------------------
            # Insert spike times at the correct indices in frame_matrix
            for i in range(N):
                t = timesteps_anim[i]
                n_nodes = spike_times.shape[0]

                for j in range(n_nodes):
                    if t in rounded[j]:
                        frame_matrix[j, i] = 1

                    #n_spikes = np.sum(np.where(t == rounded[j]))
                    #frame_matrix[j,i] = n_spikes
                    #print(n_spikes)


        self.timesteps_anim = timesteps_anim

        return self.frames, self.timesteps_anim 
                
                
               

    def animate(self, 
                e_spike_times,      # shape=(n_excitatory, spike_times) 
                i_spike_times,      
                input_spike_times, 
                output_spike_times,
                ):

        fps = 30

        fig, ax = plt.subplots()
        #self.plot_connectome(ax)

        full_conn_lines = self.conn_lines   # basis

        # Each timestep we want to take all the neurons that fired and 
        # visualize them and their synapses activating
        #dt_anim = 1/fps
        #timesteps = np.arange(self.dt_anim, self.sim_time+self.dt_anim, step=self.dt_anim) 
        

        #N = len(timesteps)
        # shape=(n_excitatory, spike_times) 
        frames, timesteps_anim = self.generate_spike_frames(e_spike_times, 
                                                            i_spike_times,      
                                                            input_spike_times, 
                                                            output_spike_times,
                                                            fps)


        N = len(timesteps_anim)
        indices = np.arange(0, N)

        def update_frame(i):

            # t = timesteps[i]
            
            print(i/N*100)
            
            ax.clear()
            self.plot_connectome(ax)
            #print(i)

            for key in self.frames:

                pos = self.positions[key]

                nodes_state = self.frames[key][0][:,int(i)]
                #print(nodes_state.shape)
                nodes_active = np.argwhere(nodes_state==1.)[:,0]
                
                xs = pos[nodes_active][:,0]
                ys = pos[nodes_active][:,1]

                ax.scatter(xs, ys, color='black')

            ax.set_xlim([-1.9, 1.9])
            ax.set_ylim([-1., 2.5])

                
            

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=850)

        ani = animation.FuncAnimation(fig, update_frame, indices)   #, fargs=(count,indices))
        ani.save("spiking_anim.mp4", writer=writer, dpi=150)



def spike_train_gen(sim_time):
    train = []
    t=0.1
    while t < sim_time:
        dt = 100 * abs(np.random.normal())
        t += dt
        if t < sim_time:
            train.append(t)
    return train


def run():

    #----------------------------------------------------------------------
    # Setting up SNN instance
    snn = SNN(n_excitatory=100, 
              n_inhibitory=30, 
              n_inputs=8, 
              n_outputs=4,
              )

    #----------------------------------------------------------------------
    # Plotting:
    fig, ax = plt.subplots()
    plotting = True
    if plotting:
        snn.plot_connectome(ax, 

                            radius_e=0.8, 
                            radius_i=0.5, 
                            
                            center_e=(0,0), 
                            center_i=(0,1.5), 

                            input_column_size=1, 
                            input_column_center=(-1.5,0.9),

                            output_column_size=1/8 * 4, 
                            output_column_center=(1.5,0.9),
                            )
    plt.savefig('test.png')

    # Dummy input spikes:
    sim_time = 10 * 1000
    inputs = [spike_train_gen(sim_time) for i in range(snn.n_inputs)]

    #----------------------------------------------------------------------
    # Simulating:
    T = 0
    sim_data = snn.simulate(input_spikes=inputs, sim_time=sim_time, T=T)
    e_spike_times = np.array(sim_data[0])
    i_spike_times = np.array(sim_data[1])
    input_spike_times = np.array(sim_data[2])
    output_spike_times = np.array(sim_data[3])
    
    #----------------------------------------------------------------------
    # Animating:
    snn.animate(e_spike_times, 
                i_spike_times, 
                input_spike_times, 
                output_spike_times)


if __name__ == "__main__":
    run()

