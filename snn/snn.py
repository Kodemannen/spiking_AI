import nest
nest.set_verbosity("M_WARNING")

import multiprocessing as mp
from joblib import delayed, Parallel
import sys

import numpy as np
#np.random.seed(13)


import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections as mc
import seaborn as sns
sns.set()

import json

from .functions import *


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
                 dt=0.1,
                 input_node_type='spike_generator'):
        """
        Main pop : consists of excitatory and inhibitory synapses
        Input synapses: only excitatory synapses
        Output pop : no synapses 
        """

        self.input_node_type = input_node_type

        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.n_total_nodes = n_excitatory + n_inhibitory + n_inputs + n_outputs

        self.dt = dt
        self.use_noise = use_noise

        nest.ResetKernel()
        nest.SetKernelStatus(dict(
                            resolution=dt,
                            ))

        
        #-----------------------------------------------
        # Loading parameters:
        #-----------------------------------------------
        with open('snn/default.json', 'r') as f:
          default_cfg = json.load(f)

        self.snn_conf = default_cfg.copy()
        if snn_config:
            self.snn_conf.update(snn_config)

        #-----------------------------------------------
        # Creating nodes:
        #-----------------------------------------------
        # Must be created first and in the correct order.
        e_lif_params = self.snn_conf["e_lif_params"]
        i_lif_params = self.snn_conf["i_lif_params"]

        self.e_population = nest.Create('iaf_psc_alpha', 
                                        n_excitatory, 
                                        e_lif_params)

        self.i_population = nest.Create('iaf_psc_alpha', 
                                        n_inhibitory, 
                                        i_lif_params)


        #-----------------------------------------------
        # Choosing format for inputs:
        #-----------------------------------------------
        if input_node_type=='spike_generator':  
            self.input_nodes = nest.Create('spike_generator', 
                                            n_inputs)

        elif input_node_type=='poisson_generator':
            self.input_nodes = nest.Create('poisson_generator', 
                                            n_inputs)


        self.output_nodes = nest.Create('iaf_psc_alpha', 
                                        n_outputs, 
                                        e_lif_params)


        #-----------------------------------------------
        # Useful groupings:
        #-----------------------------------------------
        self.all_nodes = (self.e_population 
                          + self.i_population 
                          + self.input_nodes
                          + self.output_nodes) 

        self.all_source_nodes = (self.e_population 
                                 + self.i_population 
                                 + self.input_nodes) 

        self.all_target_nodes = (self.e_population 
                                 + self.i_population 
                                 + self.output_nodes) 

        self.populations = dict(excitatory=self.e_population, 
                                inhibitory=self.i_population, 
                                input=self.input_nodes, 
                                output=self.output_nodes)

        assert (self.e_population[0] == 1), '''Nodes must be created 
                                            first and has to be in the order 
                                            exc, inh, input, output'''

        #-----------------------------------------------
        # Creating spike detectors:
        #-----------------------------------------------
        self.e_spike_detector = nest.Create('spike_detector')
        self.i_spike_detector = nest.Create('spike_detector')

        self.input_spike_detector = nest.Create('spike_detector')
        self.output_spike_detector = nest.Create('spike_detector')

        #-----------------------------------------------
        # Connection rules and parameters:
        #-----------------------------------------------
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
        #-----------------------------------------------
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

        
        # parrot neuron for input spike collection
        input_parrots = nest.Create('parrot_neuron', 
                                    n_inputs)

        #-----------------------------------------------
        # Connecting to spike detectors:
        #-----------------------------------------------
        nest.Connect(self.e_population, self.e_spike_detector)
        nest.Connect(self.i_population, self.i_spike_detector)

        nest.Connect(self.input_nodes, input_parrots)
        nest.Connect(input_parrots, self.input_spike_detector)

        nest.Connect(self.output_nodes, self.output_spike_detector)

        return None



    def set_positions(self, 
                      seed=0,

                      radius_e=0.8, 
                      radius_i=0.5, 
                      
                      center_e=(0,0), 
                      center_i=(0,2), 

                      input_column_size=1, 
                      input_column_center=(-1.5,0.9),

                      output_column_size=1, 
                      output_column_center=(1.5,0.9),
                      ):


        """
        Randomizes positions for all nodes
        """

        np.random.seed(seed)

        #-----------------------------------------------
        # Random generating main pop positions:
        #-----------------------------------------------
        positions_e = get_circular_positions(n_neurons=self.n_excitatory, 
                                             radius=radius_e, 
                                             center=center_e)

        positions_i = get_circular_positions(n_neurons=self.n_inhibitory, 
                                             radius=radius_i, 
                                             center=center_i)

        #-----------------------------------------------
        # Generating input column:
        #-----------------------------------------------
        positions_input = get_vertical_line_positions(n_neurons=self.n_inputs,
                                                      column_size=input_column_size,
                                                      column_center=input_column_center)

        #-----------------------------------------------
        # Generating output column:
        #-----------------------------------------------
        positions_output = get_vertical_line_positions(n_neurons=self.n_outputs,
                                                       column_size=output_column_size,
                                                       column_center=output_column_center)

        self.positions_e = positions_e
        self.positions_i = positions_i
        self.positions_input = positions_input
        self.positions_output = positions_output

        

        #-----------------------------------------------
        # Storing positions by population in a dictionary
        #-----------------------------------------------
        self.positions = dict(excitatory = positions_e,
                              inhibitory = positions_i,
                              input = positions_input,
                              output = positions_output)


        #-----------------------------------------------
        # storing all positions in a single box:
        #-----------------------------------------------
        self.all_positions = np.zeros((self.n_total_nodes, 2))
        self.all_positions[np.array(self.e_population)-1] = positions_e
        self.all_positions[np.array(self.i_population)-1] = positions_i
        self.all_positions[np.array(self.input_nodes)-1] = positions_input
        self.all_positions[np.array(self.output_nodes)-1] = positions_output


        return 0

    
    def get_conn_lines(self):

        """
        Function for creating lines between each pair of nodes that are connected by a synapse.
        Stores the lines in a dictionary categorized by the population that the pre-synaptic node
        belongs to, so that they can be fetched by the index of a sender.
        """

        lines_box = []
        for node in self.all_source_nodes:
            #conns.append
            conn = np.array(nest.GetConnections(source=[node], target=self.all_target_nodes))
            n = len(conn)

            if n==0:            # Dead node. Doesn't send.
                continue

            sources = conn[:,0]
            targets = conn[:,1]

            source_positions = self.all_positions[sources-1]
            target_positions = self.all_positions[targets-1]

            lines = np.zeros((n, 2, 2))
            lines[:,0] = source_positions
            lines[:,1] = target_positions

            lines_box.append(lines)

        self.lines_box = np.array(lines_box)
        
        return 0



    def plot_nodes(self, ax, indices):
        pass



    def plot_all_nodes(self, ax):
        
        for key in self.positions:
            pos = self.positions[key]
            ax.scatter(pos[:,0],
                       pos[:,1], 
                       label=key)
        return 0     


    def plot_lines(self, ax, sender_indices):
          
        for ind in sender_indices:
            lines = self.lines_box[ind]

            lc = mc.LineCollection(lines, linewidths=0.1, color='black') # choose color here
            ax.add_collection(lc) 

        return 0


    def plot_all_lines(self, ax):

        n = len(self.lines_box)
        for i in range(n):
            lines = self.lines_box[i]

            lc = mc.LineCollection(lines, linewidths=0.1) # choose color here
            ax.add_collection(lc) 

        return 0


    def plot_connectome(self, ax):
        '''
        Plots all the nodes and connections in the SNN
        '''

        #-----------------------------------------------
        # Plotting nodes:
        #-----------------------------------------------
        self.plot_all_nodes(ax)


        #-----------------------------------------------
        # Plotting connection lines:
        #-----------------------------------------------
        self.plot_all_lines(ax)


        #plt.axis('off')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.legend()

        #self.base_ax = ax
        #self.base_fig = fig
        #plt.savefig('test.png')
        #self.ax = ax

        return 0
    

    def __run_simulation(self, sim_time=100, T=0):

        #----------------------------------------------
        # simulate
        #-----------------------------------------------
        nest.Simulate(sim_time)

        #----------------------------------------------
        # analysis
        #-----------------------------------------------

        stat_e = nest.GetStatus(self.e_spike_detector, 'events')[0]
        stat_i = nest.GetStatus(self.i_spike_detector, 'events')[0]     

        stat_input = nest.GetStatus(self.input_spike_detector, 'events')[0]
        stat_output = nest.GetStatus(self.output_spike_detector, 'events')[0]

        # stat_x['times'] is a one dimensional list of spike times
        # stat_x['senders'] is a one dimensional list of gids 
        # corresponding to the spike times
        

        #----------------------------------------------
        # separating out the firings from the most 
        # recent simulation 
        #-----------------------------------------------
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
        rate_input = len(times_input) * 1000.0 / (
                sim_time * float(self.n_inputs))
        rate_output = len(times_output) * 1000.0 / (
                sim_time * float(self.n_outputs))

        print('mean excitatory rate: {0:.2f} Hz'.format(rate_e))
        print('mean inhibitory rate: {0:.2f} Hz'.format(rate_i))

        return (e_spike_times, i_spike_times, input_spike_times, 
               output_spike_times, (rate_e, rate_i, rate_output))



    def simulate(self, input_spikes, sim_time, T=0):

        self.sim_time = sim_time
        input_spike_times = [{'spike_times': np.round(spt, 1) + T } 
                                            for spt in input_spikes]

        #input_spike_times = {'spike_times': np.arange(0.1, sim_time, step=0.2)}

        nest.SetStatus(self.input_nodes, input_spike_times)

        return self.__run_simulation(sim_time, T=T)


    
    def generate_spike_frames(self, 
                              e_spike_times,      # shape=(n_excitatory, spike_times) 
                              i_spike_times,      
                              input_spike_times, 
                              output_spike_times,
                              fps=30): 
        '''
        Generates animation/plot frames with the set fps corresponding to spike times
        '''

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
            n_nodes = spike_times.shape[0]


            #----------------------------------------------
            # Rounding up to grid:
            #----------------------------------------------
            rounded = (spike_times // dt_anim) * dt_anim   # rounded to the left
            
            # Rounding to the right the ones that should be rounded to the right:
            for i in range(n_nodes):
                rest = spike_times[i] - rounded[i]
                to_add = (rest > (dt_anim/2)) * dt_anim
                rounded[i] += to_add


            #-------------------------------------------------
            # Insert spike times at the correct indices in 
            # frame_matrix
            #-------------------------------------------------
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
                
                
    def get_spikes(self, T, sim_time):
        # T = time up to
        
        stat_e = nest.GetStatus(self.e_spike_detector, 'events')[0]
        stat_i = nest.GetStatus(self.i_spike_detector, 'events')[0]     
        stat_input = nest.GetStatus(self.input_spike_detector, 'events')[0]
        stat_output = nest.GetStatus(self.output_spike_detector, 'events')[0]

        # stat_x['times'] is a one dimensional list of spike times
        # stat_x['senders'] is a one dimensional list of gids 
        # corresponding to the spike times
        

        #----------------------------------------------
        # separating out the firings from the most 
        # recent simulation 
        #-----------------------------------------------
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
        rate_output = len(times_output) * 1000.0 / (
                sim_time * float(self.n_outputs))

        print('mean excitatory rate: {0:.2f} Hz'.format(rate_e))
        print('mean inhibitory rate: {0:.2f} Hz'.format(rate_i))

        return (e_spike_times, i_spike_times, input_spike_times, 
               output_spike_times, (rate_e, rate_i, rate_output))
               

    def animate(self, 
                e_spike_times,      # shape=(n_excitatory, spike_times) 
                i_spike_times,      
                input_spike_times, 
                output_spike_times,
                ):

        fps = 30

        fig, ax = plt.subplots()
        #self.plot_connectome(ax)

        #full_conn_lines = self.conn_lines   # basis

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
                pop = self.populations[key]

                nodes_state = self.frames[key][0][:,int(i)]
                #print(nodes_state.shape)
                nodes_active = np.argwhere(nodes_state==1.)[:,0]
                
                if len(nodes_active) > 0:

                    xs = pos[nodes_active][:,0]
                    ys = pos[nodes_active][:,1]

                    ax.scatter(xs, ys, color='black')
                    
                    if not key=='output':
                        #------------------------------------
                        # Lines:
                        indices = nodes_active + pop[0] - 1
                        self.plot_lines(ax, sender_indices=indices)
                    

            ax.set_xlim([-1.9, 1.9])
            ax.set_ylim([-1., 2.5])


        #------------------------------------------------------------------
        # Generating .mp4
        #------------------------------------------------------------------
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=850)

        ani = animation.FuncAnimation(fig, update_frame, indices)   #, fargs=(count,indices))
        ani.save("spiking_anim.mp4", writer=writer, dpi=150)




    def plot_one_iteration(self,
                           e_spike_times,      # shape=(n_excitatory, spike_times) 
                           i_spike_times,      
                           input_spike_times, 
                           output_spike_times,
                            ):

        pass


def run():

    #----------------------------------------------------------------------
    # Setting up SNN instance
    #----------------------------------------------------------------------
    snn = SNN(n_excitatory=15, 
              n_inhibitory=5, 
              n_inputs=6, 
              n_outputs=2,
              )

    #----------------------------------------------------------------------
    # Initializing positions:
    #----------------------------------------------------------------------
    snn.set_positions(seed=2,#seed=np.random.randint(low=0,high=10e7),

                      radius_e=0.8, 
                      radius_i=0.5, 
                        
                      center_e=(0,0), 
                      center_i=(0,1.5), 

                      input_column_size=1, 
                      input_column_center=(-1.5,0.9),

                      output_column_size=1/8 * 4, 
                      output_column_center=(1.5,0.9),
                      )
    snn.get_conn_lines()

    #----------------------------------------------------------------------
    # Plotting connectome as image:
    #----------------------------------------------------------------------
    plotting = True
    if plotting:
        fig, ax = plt.subplots()
        snn.plot_connectome(ax)
        #plt.legend(loc=4)
        plt.savefig('test.png')



    #----------------------------------------------------------------------
    # Simulating:
    #----------------------------------------------------------------------
    sim_time = 10 * 1000
    
    # Dummy input spikes:
    inputs = [spike_train_gen(sim_time) for i in range(snn.n_inputs)]

    T = 0
    sim_data = snn.simulate(input_spikes=inputs, sim_time=sim_time, T=T)
    e_spike_times = np.array(sim_data[0])
    i_spike_times = np.array(sim_data[1])
    input_spike_times = np.array(sim_data[2])
    output_spike_times = np.array(sim_data[3])
    
    #----------------------------------------------------------------------
    # Animating activity:
    #----------------------------------------------------------------------
    snn.animate(e_spike_times, 
                i_spike_times, 
                input_spike_times, 
                output_spike_times)


if __name__ == "__main__":
    run()

