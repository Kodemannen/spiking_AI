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
        target = (*self.e_population, *self.i_population)
        source = (*self.e_population, *self.i_population)
        connections_main = nest.GetConnections(target=target, source=source)

        # connections_..[i] is connection number i in the list of connections in the network?
        # format: (source-gid, target-gid, target-thread, synapse-id, port)

        n_connections = len(connections_main)
        n_total_neurons = self.n_excitatory + self.n_inhibitory

        root = round(np.sqrt(n_total_neurons))

        #-----------------------------------------------
        # Plot grid style:
        # Set up in a grid after GIDs:

        layer = tp.CreateLayer(dict(
            rows = 5,
            columns = 5,
            elements = "iaf_psc_alpha",
            center = [1,2]
        ))

        print(nest.GetStatus(layer))

        return None



def test():

    snn = SNN()
    snn.plot_connectome()




if __name__ == "__main__":
    test()
