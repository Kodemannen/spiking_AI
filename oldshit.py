import matplotlib.pyplot as plt
import multiprocessing as mp
import nest
nest.set_verbosity("M_WARNING")
import numpy as np
from joblib import delayed, Parallel
import json
import numpy as np
import nest.topology as topp
import nest
import seaborn as sns
import sys
# np.random.seed(eval(sys.argv[1]))
np.random.seed(10)
sns.set()



def Plot_connectome(population, positions, connections, ax):

    #-----------------------------------------------
    # Plotting connections:
    pop_size = len(population)
    pop_indices = np.arange(pop_size)
    pop_gids = np.arange(population[0], population[-1]+1)

    n_connections = len(connections)

    for i in range(n_connections):

        synapse = connections[i] # Synapse number i in the network.
        # shape = (source-gid, target-gid, target-thread, synapse-id, port)

        gid_sender = synapse[0]
        gid_receiver = synapse[1]
        print(gid_sender)
        print(gid_receiver)

        index_sender = np.argwhere((gid_sender)==pop_gids) # index in population
        index_receiver = np.argwhere((gid_receiver)==pop_gids)
        print(index_sender)
        print(index_receiver)
        print(pop_gids)
        print(len(pop_gids))
        print("horeslut")

        index_sender = np.argwhere(gid_sender==pop_gids)[0,0]
        index_receiver = np.argwhere(gid_receiver==pop_gids)[0,0]

        sender_position = positions[index_sender]
        receiver_position = positions[index_receiver]

        xs = [sender_position[0], receiver_position[0]] # The x-coordinates of the pair of neurons
        ys = [sender_position[1], receiver_position[1]] # Ditto but for the y-coordinates

        ax.plot(xs,ys, linewidth=0.1, alpha=0.2, color="grey")


    # Plotting neurons:
    ax.scatter(*zip(*positions), alpha=.7, color="grey")

    return None



def Create_population_positions(N, distribution):
    # distribution[i] is a list [j,k], representing that neuron i has position (j,k)

    if distribution.lower()=="uniform":
        positions = np.random.uniform(low=0, high=1, size=(N, 2))

    elif distribution.lower()=="gaussian":
        positions = np.random.normal(loc=0, scale=1, size=(N, 2))

    elif distribution.lower()=="column":
        positions = np.zeros(shape=(N,2))
        positions[:,1] = np.linspace(-1,1,N)

    return positions



def rush_b():

    #-----------------------------------------------
    # Hyperparameters:
    simtime = 100.0 # ms
    N_main = 100 # number of neurons
    N_sensory = 20
    plot = True
    epsilon = 0.1
    indegree = 10
    outdegree = 30
    sensory_rate = 10.0 # Hz?

    nest.SetKernelStatus({"overwrite_files": True})


    #-----------------------------------------------
    # Creating network and experimental tools:

    # Neuron populations:
    population_main    = nest.Create("iaf_psc_delta", N_main)
    population_sensory = nest.Create("parrot_neuron", N_sensory)
    poisson_generator  = nest.Create("poisson_generator")

    spike_detector = nest.Create("spike_detector", params={"to_file":True,"label":"spike_times.txt"})

    # recurrent connections in main pop:
    nest.Connect(population_main, population_main,
                conn_spec={"rule": "fixed_indegree", "indegree": indegree})

    # connecting sensory pop to main pop:
    nest.Connect(population_sensory, population_main,
                 conn_spec={"rule": "fixed_outdegree", "outdegree": outdegree})

    # Fetching connectons for plotting:
    connections_main    = nest.GetConnections(source=population_main, target=population_main)
    connections_sensory = nest.GetConnections(source=population_sensory)
    # connections_..[i] is connection number i in the list of connections in the network?
    # shape of element connections_..[i] = (source-gid, target-gid, target-thread, synapse-id, port)

    # Connecting input stimulus:
    nest.Connect(poisson_generator, population_sensory)

    # Connecting spike detector to detect spikes:
    nest.Connect(population_sensory, spike_detector)



    #-----------------------------------------------
    # Setting up positions:
    positions_main    = Create_population_positions(N_main, distribution="gaussian")
    positions_sensory = Create_population_positions(N_sensory, distribution="column")


    nest.SetStatus(poisson_generator, {"rate": sensory_rate})


    nest.Simulate(simtime)

    events = nest.GetStatus(spike_detector)[0]["events"]
    senders = events["senders"]
    times = events["times"]
    #plt.scatter(times,senders)
    #plt.show()

    if plot==True:
        fig, ax = plt.subplots()
        Plot_connectome(population_main, positions_main, connections_main, ax)
        print("hore")
        Plot_connectome(population_sensory, positions_sensory, connections_sensory, ax)
        fig.savefig("fig.pdf")



def test():

    snn = SNN()




if __name__ == "__main__":
    test()
