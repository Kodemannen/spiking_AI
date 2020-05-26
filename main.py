import matplotlib.pyplot as plt
import numpy as np
import nest.topology as topp
import nest
import seaborn as sns
import sys
# np.random.seed(eval(sys.argv[1]))
np.random.seed(10)

def Plot_connectome(population, positions, connections, ax="none", save="no"):

    if ax=="none":
        sns.set()
        fig, ax = plt.subplots()

    # Plotting connections:
    n = len(connections)
    for i in range(n):

        synapse = connections[i] # Synapse number i in the network. (source-gid, target-gid, target-thread, synapse-id, port)
        # positions[synapse[i][0]] is the position of the sender, positions[synapse[i][1]]

        sender_position = positions[synapse[0]-1]
        receiver_position = positions[synapse[1]-1]

        xs = [sender_position[0], receiver_position[0]]
        ys = [sender_position[1], receiver_position[1]]

        ax.plot(xs,ys, linewidth=0.1, alpha=0.2, color="grey")

    # plot neurons:
    ax.scatter(*zip(*positions), alpha=.7, color="grey")

    #ax.axis("off")
    if save=="yes":
        fig.savefig("fig.pdf")
    #plt.show()



def Create_population_positions(N, distribution):

    if distribution=="uniform":
        positions = np.random.uniform(low=0, high=1, size=(N, 2)) # the i'th list [j,k] means neuron i had position (j,k)
    elif distribution.lower()=="gaussian":
        positions = np.random.normal(loc=0, scale=1, size=(N, 2)) # the i'th list [j,k] means neuron i had position (j,k)
    return 0



if __name__ == "__main__":

    simtime = 100.0 # ms
    N = 100 # number of neurons
    N_sensory = 20
    plot = True
    epsilon = 0.1
    indegree = 0
    sensory_rate = 10.0 # Hz?

    nest.SetKernelStatus({"overwrite_files": True})




      ############################################
     # Creating network and experimental tools: #
    ############################################

    # Neuron populations:
    main_population = nest.Create("iaf_psc_delta", N)
    sensory_pop = nest.Create("parrot_neuron", N_sensory)
    poisson_generator = nest.Create("poisson_generator")

    # Device needed for detecting the spikes of interest:
    spike_detector = nest.Create("spike_detector", params={"to_file":True,"label":"spike_times.txt"})





      ###########################
     # Setting up connections: #
    ###########################
    nest.Connect(sensory_pop, spike_detector)
    nest.Connect(main_population, main_population, conn_spec={"rule": "fixed_indegree", "indegree": indegree})
    nest.Connect(poisson_generator, sensory_pop)

    connections_main_pop = nest.GetConnections(target=main_population) # Element connections[i] is connection number i in the list of connections in the network? Can be given with an argument source=arg or target=arg. (source-gid, target-gid, target-thread, synapse-id, port)



     #######################
    # Setting up positions: #

    nest.SetStatus(poisson_generator, {"rate": sensory_rate})




    nest.Simulate(simtime)

    events = nest.GetStatus(spike_detector)[0]["events"]
    senders = events["senders"]
    times = events["times"]
    #plt.scatter(times,senders)
    #plt.show()

    if plot==True:
        Plot_connectome(main_population, positions, connections_main_pop, save="yes")

