import multiprocessing as mp
from joblib import delayed, Parallel
import sys

import numpy as np
np.random.seed(10)

import nest
import nest.topology as tp
nest.set_verbosity("M_WARNING")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import json


def get_conn_pairs(conn_ij):

    n_connections = len(conn_ij)
    connection_pairs = np.zeros((n_connections, 2), dtype=np.int)
    for i in range(n_connections):
        source = connections_main[i][0]
        target = connections_main[i][1]
        connection_pairs[i] = source, target

    return connection_pairs



class SNN:

    def __init__(self, snn_config=None, n_excitatory=3, n_inhibitory=2, use_noise=False):
        """
        Main pop: consists of excitatory and inhibitory neurons
        Input pop: only excitatory
        """

        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.use_noise = use_noise

        nest.ResetKernel()

        #-----------------------------------------------
        # Fetching config parameters from file:
        with open('default.json', 'r') as f:
          default_cfg = json.load(f)

        self.snn_conf = default_cfg.copy()
        if snn_config:
            self.snn_conf.update(snn_config)

        #-----------------------------------------------
        # Creating main population:
        e_lif_params = self.snn_conf["e_lif_params"]
        i_lif_params = self.snn_conf["i_lif_params"]
        self.e_population = nest.Create('iaf_psc_exp', n_excitatory, e_lif_params)
        self.i_population = nest.Create('iaf_psc_exp', n_inhibitory, i_lif_params)

        #-----------------------------------------------
        # Creating noise generator, if included:
        if self.use_noise:
            noise_gen_params = self.snn_conf["noise_gen_params"]
            noise_generator = nest.Create('poisson_generator', 1, noise_gen_params)

        #-----------------------------------------------
        # Creating spike detectors:
        self.e_spike_detector = nest.Create('spike_detector')
        self.i_spike_detector = nest.Create('spike_detector')

        #-----------------------------------------------
        # Fetching connection rules and parameters:
        syn_spec_ee = self.snn_conf["syn_spec_ee"]
        syn_spec_ei = self.snn_conf["syn_spec_ei"]
        syn_spec_ie = self.snn_conf["syn_spec_ie"]
        syn_spec_ii = self.snn_conf["syn_spec_ii"]
        syn_spec_noise = self.snn_conf["syn_spec_noise"]
        rule_dict_exc = self.snn_conf["rule_dict_exc"]
        rule_dict_inh = self.snn_conf["rule_dict_inh"]
        rule_dict_noise = self.snn_conf["rule_dict_noise"]

        #-----------------------------------------------
        # Connecting noise to main pop:
        if self.use_noise:
            nest.Connect(noise_generator, self.e_population, rule_dict_noise, syn_spec_noise)
            nest.Connect(noise_generator, self.i_population, rule_dict_noise, syn_spec_noise)

        #-----------------------------------------------
        # Connecting recurrent connections in the main pop:
        nest.Connect(self.e_population, self.e_population, rule_dict_exc, syn_spec_ee)
        nest.Connect(self.e_population, self.i_population, rule_dict_exc, syn_spec_ei)
        nest.Connect(self.i_population, self.i_population, rule_dict_inh, syn_spec_ii)
        nest.Connect(self.i_population, self.e_population, rule_dict_inh, syn_spec_ie)

        #-----------------------------------------------
        # Connecting main pop to spike detectors:
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
    

        conn_pairs = dict()
        positions = dict()
        #-----------------------------------------------
        # Collection connection pairs in an array
        # Need to keep track of which neuron has which position
        for key in conn_groups:
            conn_ij = conn_groups[key]
            n_neurons = len(conn_ij)

            conn_pairs[key] = get_conn_pairs(conn_groups[key])



        
        #-----------------------------------------------
        # Random generating positions:
        n_total_neurons = self.n_excitatory + self.n_inhibitory
        positions_excitatory = np.random.randint(low=0, high=10, size=(self.n_excitatory, 2))
        positions_inhibitory = np.random.randint(low=10, high=20, size=(self.n_inhibitory, 2))


        #-----------------------------------------------
        # Plotting neurons:
        plt.scatter(positions_excitatory[:,0], positions_excitatory[:,1], label='Excitatory')
        plt.scatter(positions_inhibitory[:,0], positions_inhibitory[:,1], label='Inhibitory')

        # Plotting connections:
        for i in range(n_connections):
            conn_pair 
            plt.plot(

        #plt.axis('off')
        plt.savefig('test.png')


        return None



def test():

    snn = SNN()
    snn.plot_connectome()




if __name__ == "__main__":
    test()
