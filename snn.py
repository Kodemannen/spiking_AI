import multiprocessing as mp
from joblib import delayed, Parallel
import sys

import numpy as np
np.random.seed(13)

import nest
import nest.topology as tp
nest.set_verbosity("M_WARNING")

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns
sns.set()

import json

from functions import *


class SNN:
    """
    Builds, simulates and visualizes a spiking neural network 
    """

    def __init__(self, 
                 snn_config=None, 
                 n_excitatory=100, 
                 n_inhibitory=25, 
                 n_inputs=10, 
                 n_outputs=10, 
                 use_noise=False
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
        self.use_noise = use_noise

        nest.ResetKernel()

        
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

        self.e_population = nest.Create('iaf_psc_alpha', n_excitatory, e_lif_params)
        self.i_population = nest.Create('iaf_psc_alpha', n_inhibitory, i_lif_params)
        self.input_nodes = nest.Create('iaf_psc_alpha', n_inputs, e_lif_params)
        self.output_nodes = nest.Create('iaf_psc_alpha', n_outputs, e_lif_params)

        assert (self.e_population[0] == 1), '''Nodes must be created first and has to be in the order 
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
        nest.Connect(self.e_population, self.e_population, rule_dict_e, syn_spec_ee)
        nest.Connect(self.e_population, self.i_population, rule_dict_e, syn_spec_ei)
        nest.Connect(self.i_population, self.i_population, rule_dict_i, syn_spec_ii)
        nest.Connect(self.i_population, self.e_population, rule_dict_i, syn_spec_ie)
        nest.Connect(self.input_nodes, self.e_population, rule_dict_inp, syn_spec_inp)
        nest.Connect(self.e_population, self.output_nodes, rule_dict_output, syn_spec_ee)

        #-----------------------------------------------
        # Connecting to spike detectors:
        nest.Connect(self.e_population, self.e_spike_detector)
        nest.Connect(self.i_population, self.i_spike_detector)
        nest.Connect(self.input_nodes, self.input_spike_detector)
        nest.Connect(self.output_nodes, self.output_spike_detector)

        return None


    def plot_connectome(self, 
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

        #connections_.. = nest.GetConnections(target=target, source=source)   
        # format: (source-gid, target-gid, target-thread, synapse-id, port)
        # For an individual synapse we get:
            # source = connections_..[i][0]
            # target = connections_..[i][1]

        conn_groups = dict(
            conn_ee = nest.GetConnections(source=self.e_population, target=self.e_population),
            conn_ei = nest.GetConnections(source=self.e_population, target=self.i_population),
            conn_ii = nest.GetConnections(source=self.i_population, target=self.i_population),
            conn_ie = nest.GetConnections(source=self.i_population, target=self.e_population),
            conn_inp = nest.GetConnections(source=self.input_nodes, target=self.e_population),
            conn_output = nest.GetConnections(source=self.e_population, target=self.output_nodes),
        )
    
        #-----------------------------------------------
        # Random generating main pop positions:
        positions_e = get_circular_positions(n_neurons=self.n_excitatory, radius=radius_e, center=center_e)
        positions_i = get_circular_positions(n_neurons=self.n_inhibitory, radius=radius_i, center=center_i)

            # The relationship between neuron GID and position is: 

                # For excitatory:
                    # position index = GID - 1

                # For inhibitory:
                    # position index = GID - 1 - self.n_excitatory

        # Generating input column:
        positions_inp = get_vertical_line_positions(
                                                    n_neurons=self.n_inputs,
                                                    column_size=input_column_size,
                                                    column_center=input_column_center
                                                    )
        # Generating output column:
        positions_output = get_vertical_line_positions(
                                                    n_neurons=self.n_outputs,
                                                    column_size=output_column_size,
                                                    column_center=output_column_center
                                                    )

        #-------------------Plotting--------------------
        fig, ax = plt.subplots()

        #-----------------------------------------------
        # Collecting connection pairs in an array
        conn_pairs = dict()
        for key in conn_groups:
            conn_ij = conn_groups[key]
            conn_pairs[key] = get_conn_pairs(conn_groups[key])    

        #-----------------------------------------------
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

                if sender_type=='e':
                    source_pos_index = pair[0]-1
                    source_pos = positions_e[source_pos_index]

                elif sender_type=='i':
                    source_pos_index = pair[0]-1 - self.n_excitatory
                    source_pos = positions_i[source_pos_index]

                elif sender_type=='input': 
                    source_pos_index = pair[0]-1 - self.n_excitatory - self.n_inhibitory 
                    source_pos = positions_inp[source_pos_index]

                if receiver_type=='e':
                    receiver_pos_index = pair[1]-1
                    receiver_pos = positions_e[receiver_pos_index]

                elif receiver_type=='i':
                    receiver_pos_index = pair[1]-1 -self.n_excitatory
                    receiver_pos = positions_i[receiver_pos_index]

                elif receiver_type=='output':
                    receiver_pos_index = pair[1]-1 - self.n_excitatory - self.n_inhibitory - self.n_inputs
                    receiver_pos = positions_output[receiver_pos_index]

                lines[i, 0] = source_pos 
                lines[i, 1] = receiver_pos


            conn_lines[key] = lines
            lc = mc.LineCollection(lines, linewidths=0.1)       # choose color here
            ax.add_collection(lc) 
            #_______________________________________________
            

        # Plotting nodes:
        ax.scatter(positions_e[:,0], positions_e[:,1], label='Excitatory')
        ax.scatter(positions_i[:,0], positions_i[:,1], label='Inhibitory')
        ax.scatter(positions_inp[:,0], positions_inp[:,1], color='grey', label='Input')
        ax.scatter(positions_output[:,0], positions_output[:,1], color='grey', label='Output')

        # plt.axis('off')
        ax.set_aspect('equal')
        #ax.legend()

        self.ax = ax
        self.fig = fig
        plt.savefig('test.png')

        return None
    

    def simulate(self, sim_time):
        pass


def test():

    snn = SNN(n_excitatory=100, n_inhibitory=30)
    snn.plot_connectome()




if __name__ == "__main__":
    test()
