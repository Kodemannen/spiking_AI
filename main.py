import numpy as np
import nest


def Run_simulation(input):
    
    #pylint: disable=unused-argument
    N = 1000
    N_inp = 100

    main_nodes = nest.Create("iaf_psc_delta", N)
    inp_nodes = nest.Create("iaf_psc_delta", N_inp)

    ##############
    # Connecting #
    nest.Connect(main_nodes, main_nodes,)