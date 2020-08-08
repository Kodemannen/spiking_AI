import numpy as np


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
    # Not sure why it doesnt work
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
