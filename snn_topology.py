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


def get_circular_positions(n_neurons, radius=1, center=(0,0)):

    # need key word arguments radius (max radius)
    r = np.random.uniform(low=-radius, high=radius, size=n_neurons)
    theta = np.random.uniform(low=0, high=2*np.pi, size=n_neurons)

    positions = np.zeros(shape=(n_neurons, 2))
    positions[:,0] = r*np.cos(theta)
    positions[:,1] = r*np.sin(theta)

    # addind center position:
    positions += center

    return positions


def get_circular_positions_uniform(n_neurons, radius=1, center=(0,0)):
    # UNFINISHED
    density = n_neurons / (2*np.pi*radius**2)   # N/area
    
    n_per_radius = round( density*radius )
    dr = radius / n_per_radius
    points_along_radius = np.arange(dr, radius+dr, step=dr)

    connection_pairs = np.zeros(shape=(n_neurons, 2))

    low_ind = 0
    high_ind = 0
    n_accumulated = 0
    for i in range(n_per_radius):
        # generate a columns with nodes and bend it around in a circle

        r = points_along_radius[i]
        n = int(round( r*density ))         # neurons in the column
        n_accumulated += n

        if n_accumulated > n_neurons:
            n

        dtheta = 2*np.pi / n                # angular step size
        thetas = np.arange(0, 2*np.pi, step=dtheta)

        print(n)
        print(thetas.shape)
        xs = r*np.cos(thetas)            
        ys = r*np.sin(thetas)

        print('xs', xs.shape)

        high_ind += n
        if high_ind > (n_neurons-1):
            high_ind = n_neurons-1
        
        print('mat', connection_pairs[low_ind:high_ind, 0].shape)  
        connection_pairs[low_ind:high_ind, 0] = xs
        connection_pairs[low_ind:high_ind, 1] = ys

        low_ind = high_ind
        print('---------------------------------------------------------')
        
    return connection_pairs 

def get_vertical_line_positions(n_neurons, column_size=1, column_center=(-1,0)):
        
    x_positions = np.zeros(shape=(n_neurons)) + column_center[0]
    y_positions = np.linspace(-column_size/2, column_size/2, n_neurons) + column_center[1]

    positions = np.array( [x_positions, y_positions] ).T
    return positions


class SNN:

    def __init__(self, snn_config=None, n_excitatory=100, n_inhibitory=25, n_inputs=10, use_noise=False):
        """
        Main pop: consists of excitatory and inhibitory neurons
        Input pop: only excitatory
        """

        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.n_inputs = n_inputs
        self.use_noise = use_noise

        nest.ResetKernel()

        #-----------------------------------------------
        # Creating nodes:
        self.e_pop_dict = 
        self.e_population = tp.CreateLayer('iaf_psc_alpha', n_excitatory)

        self.i_population = tp.Create('iaf_psc_alpha', n_inhibitory)
        self.input_nodes = tp.Create('iaf_psc_alpha', n_inputs)

        #-----------------------------------------------
        # Creating spike detectors:
        self.e_spike_detector = nest.Create('spike_detector')
        self.i_spike_detector = nest.Create('spike_detector')
        self.inp_spike_detector = nest.Create('spike_detector')

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

        # temporary:
        syn_spec_inp = syn_spec_ee
        rule_dict_inp = rule_dict_e

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

        nest.Connect(self.input_nodes, self.e_population, rule_dict_inp, syn_spec_inp)
        

        return None


    def plot_connectome(self, 
                        radius_e=1, 
                        radius_i=0.5, 
                        
                        center_e=(0,0), 
                        center_i=(0,2), 

                        input_colum_size=1, 
                        input_column_center=(-1,0)
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
            conn_ie = nest.GetConnections(source=self.i_population, target=self.e_population)
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
        positions_inp = get_vertical_line_positions(n_neurons=self.n_inputs)

        #-----------------------------------------------
        # Collecting connection pairs in an array
        conn_pairs = dict()
        for key in conn_groups:
            conn_ij = conn_groups[key]
            conn_pairs[key] = get_conn_pairs(conn_groups[key])    

        fig, ax = plt.subplots()

        #-----------------------------------------------
        # Plotting nodes:
        ax.scatter(positions_e[:,0], positions_e[:,1], label='Excitatory')
        ax.scatter(positions_i[:,0], positions_i[:,1], label='Inhibitory')

        #-----------------------------------------------
        # Plotting connection lines:
        conn_lines = dict()
        for key in conn_pairs:
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
                    source_pos_index = pair[0]-1-self.n_excitatory
                    source_pos = positions_i[source_pos_index]

                if receiver_type=='e':
                    receiver_pos_index = pair[1]-1
                    receiver_pos = positions_e[receiver_pos_index]

                elif receiver_type=='i':
                    receiver_pos_index = pair[1]-1-self.n_excitatory
                    receiver_pos = positions_i[receiver_pos_index]

                lines[i, 0] = source_pos 
                lines[i, 1] = receiver_pos

            conn_lines[key] = lines
            lc = mc.LineCollection(lines, linewidths=0.1)
            ax.add_collection(lc) 
            #_______________________________________________
            

        # plt.axis('off')
        ax.set_aspect('equal')

        self.ax = ax
        self.fig = fig
        plt.savefig('test.png')


        return None
    


def test():

    snn = SNN(n_excitatory=100, n_inhibitory=30)
    snn.plot_connectome()




if __name__ == "__main__":
    test()
