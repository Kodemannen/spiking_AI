import numpy as np

spike_data = np.load('spike_data.npy', allow_pickle=True)
e_spike_times, i_spike_times, input_spike_times, output_spike_times, _ = spike_data


print(e_spike_times)

