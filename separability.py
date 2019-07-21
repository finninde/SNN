import numpy as np
import matplotlib.pyplot as plt

from network import Network, Neuron

if __name__ == "__main__":
    '''
    Set up network
    '''
    simulation_steps = 4
    connections = 3
    seed = 81549300
    n_neurons = 5
    activity = []
    network = Network(seed, connections, n_neurons)

    '''
    Run network for initial steps
    '''
    for i in range(0, 9):
        activity.append(network.step())
        #print(activity[i].size)
    '''
    Input different stimuli
    '''
    u1 = [0.0, 0.0, 0.0, 0.0]
    activity_u1 = []
    for stimuli in u1:
        network.add_stimuli(stimuli)
        activity_u1.append(network.step())

    '''
    Set up new network
    '''
    simulation_steps = 4
    connections = 3
    seed = 81549300
    n_neurons = 5
    activity2 = []
    network2 = Network(seed, connections, n_neurons)

    '''
        Run network for initial steps
    '''
    for i in range(0, 9):
        activity2.append(network2.step())
        #print(activity2[i].size)
    '''
    input stimuli
    '''
    u2 = [1.0, 1.0, 1.0, 1.0]
    #u2 = [0.0, 0.0, 0.0, 0.0]
    activity_u2 = []
    for stimuli in u2:
        network2.add_stimuli(stimuli)
        activity_u2.append(network2.step())

    '''
    Measure distance in stimuli
    '''
    hamming_distance_u1_u2 = []

    for timestep in range(0, len(activity_u1)-1):
        for neuron in range(0, n_neurons):
            if activity_u1[timestep][neuron] != activity_u1[timestep][neuron]:
                print("som")
