import multiprocessing as mp
from joblib import delayed, Parallel
import sys

import numpy as np
np.random.seed(10)

import nest
import nest.topology as tp
nest.set_verbosity("M_WARNING")

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns
sns.set()

import json


def get_conn_pairs(conn_ij):

    n_connections = len(conn_ij)
    connection_pairs = np.zeros((n_connections, 2), dtype=np.int)
    for i in range(n_connections):
        source = conn_ij[i][0]
        target = conn_ij[i][1]
        connection_pairs[i] = source, target

    return connection_pairs



class SNN:

    def __init__(self, snn_config=None, n_excitatory=100, n_inhibitory=25, use_noise=False):
        """
        Main pop: consists of excitatory and inhibitory neurons
        Input pop: only excitatory
        """

        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.use_noise = use_noise

        nest.ResetKernel()

        #-----------------------------------------------
        # Creating main population:
        # Important that these are the first elements created, since the indices will be used for positions
        self.e_population = nest.Create('iaf_psc_alpha', n_excitatory)
        self.i_population = nest.Create('iaf_psc_alpha', n_inhibitory)

        #-----------------------------------------------
        # Creating spike detectors:
        self.e_spike_detector = nest.Create('spike_detector')
        self.i_spike_detector = nest.Create('spike_detector')

        #-----------------------------------------------
        # Connection rules and parameters:
        weight_e = 10.
        weight_i = -2.5
        outdegree = 2
        delay = 1.
        
        syn_spec_ee = dict(weight=weight_e, delay=delay)
        syn_spec_ei = dict(weight=weight_e, delay=delay)
        syn_spec_ie = dict(weight=weight_i, delay=delay)
        syn_spec_ii = dict(weight=weight_i, delay=delay)

        rule_dict_e = dict(rule="fixed_outdegree", outdegree=outdegree)
        rule_dict_i = dict(rule="fixed_outdegree", outdegree=outdegree)


        #-----------------------------------------------
        # Connecting recurrent connections:
        nest.Connect(self.e_population, self.e_population, rule_dict_e, syn_spec_ee)
        nest.Connect(self.e_population, self.i_population, rule_dict_e, syn_spec_ei)
        nest.Connect(self.i_population, self.i_population, rule_dict_i, syn_spec_ii)
        nest.Connect(self.i_population, self.e_population, rule_dict_i, syn_spec_ie)

        #-----------------------------------------------
        # Connecting to spike detectors:
        nest.Connect(self.e_population, self.e_spike_detector)
        nest.Connect(self.i_population, self.i_spike_detector)

        return None


    def plot_connectome(self):

        #-----------------------------------------------
        # Getting connections:
        #target = (*self.e_population, *self.i_population)
        #source = (*self.e_population, *self.i_population)

        #connections_.. = nest.GetConnections(target=target, source=source)   # format: (source-gid, target-gid, target-thread, synapse-id, port)
        # For an individual synapse we get:
            # source = connections_..[i][0]
            # target = connections_..[i][1]

        conn_groups = dict(
            conn_ee = nest.GetConnections(source=self.e_population, target=self.e_population),
            conn_ei = nest.GetConnections(source=self.e_population, target=self.i_population),
            conn_ii = nest.GetConnections(source=self.i_population, target=self.i_population),
            conn_ie = nest.GetConnections(source=self.i_population, target=self.e_population)
        )
    
        #-----------------------------------------------
        # Random generating positions:
        positions_excitatory = np.random.randint(low=0, high=10, size=(self.n_excitatory, 2))
        positions_inhibitory = np.random.randint(low=10, high=20, size=(self.n_inhibitory, 2))
        # The relationship between neuron GID and position is: 
            # For excitatory:
                # position index = GID - 1
            # For inhibitory:
                # position index = GID - 1 - self.n_excitatory


        #-----------------------------------------------
        # Collecting connection pairs in an array
        conn_pairs = dict()
        for key in conn_groups:
            conn_ij = conn_groups[key]
            conn_pairs[key] = get_conn_pairs(conn_groups[key])    # all the connection pairs from pop i to pop j


        fig, ax = plt.subplots()

        #-----------------------------------------------
        # Plotting neurons:
        ax.scatter(positions_excitatory[:,0], positions_excitatory[:,1], label='Excitatory')
        ax.scatter(positions_inhibitory[:,0], positions_inhibitory[:,1], label='Inhibitory')

        # Plotting connections:
        conn_lines = dict()
        for key in conn_pairs:
            print(key)
            print('---------------------------------------------------------')
            sender_type = key[-2]
            receiver_type = key[-1]
            
            pairs = conn_pairs[key]
            n_pairs = len(pairs)
            lines = np.zeros(shape=(n_pairs, 2, 2))
            
            for i in range(n_pairs):
                pair = pairs[i]

                if sender_type=='e':
                    source_pos_index = pair[0]-1
                    source_pos = positions_excitatory[source_pos_index]

                elif sender_type=='i':
                    source_pos_index = pair[0]-1-self.n_excitatory
                    source_pos = positions_inhibitory[source_pos_index]

                if receiver_type=='e':
                    receiver_pos_index = pair[1]-1
                    receiver_pos = positions_excitatory[receiver_pos_index]

                elif receiver_type=='i':
                    receiver_pos_index = pair[1]-1-self.n_excitatory
                    print(receiver_pos_index)
                    receiver_pos = positions_inhibitory[receiver_pos_index]

                lines[i, 0] = source_pos 
                lines[i, 1] = receiver_pos

                #source_pos_index = pair[0]-1 if sender_type=='e' else pair[0]-1-self.n_excitatory
                #target_pos_index = pair[1]-1 if receiver_type=='e' else pair[1]-1-self.n_excitatory
                #
                #source_pos = positions_excitatory 
            
            conn_lines[key] = lines
            lc = mc.LineCollection(lines, linewidths=0.1)
            ax.add_collection(lc) 
            

        #plt.axis('off')
        plt.savefig('test.png')


        return None



def test():

    snn = SNN()
    snn.plot_connectome()




if __name__ == "__main__":
    test()
