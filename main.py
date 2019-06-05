import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy.stats import entropy


class Neuron:
    def __init__(self, max_connections, tiring_rate):
        self.outputs = [] 
        self.inputs = []
        self.inp = 0
        self.firing_probability = 0.4
        self.tiredness = 0.2
        self.active_old = np.random.randint(0, 2)
        self.active = 0
        self.first_run = True
        self.max_connections = max_connections
        self.tiring_rate = tiring_rate

    def update(self):
        if self.first_run:
            if self.active_old:
                self.tiredness = np.clip(self.tiredness + self.tiring_rate, 0, 1)
                self.first_run = False
            else:
                self.active_old = 0
                self.tiredness = np.clip(self.tiredness - self.tiring_rate, 0, 1)
                self.first_run = False          
        else:
            self.threshold()
            if self.active_old:
                self.tiredness = np.clip(self.tiredness + self.tiring_rate, 0, 1)
            else:
                self.tiredness = np.clip(self.tiredness - self.tiring_rate, 0, 1)
        
    def threshold(self):
        #TODO: se mer på tiredness
        for neuron in self.inputs:
            if neuron.active_old:
                self.inp = self.inp + 1
        if self.inp == 0:
            prob_now = np.clip(self.firing_probability - self.tiredness, 0, 1)
        else:
            prob_now = np.clip(self.firing_probability + (self.inp/(2*(self.max_connections+1))) - self.tiredness, 0, 1)
        self.inp = 0
        if random.random() < prob_now:
            self.active_old = 1
        else:
            self.active_old = 0


def update_state(neurons):
    for neuron in neurons:
        neuron.update()


def active_neurons(neurons):
    active = 0
    for neuron in neurons:
        if neuron.active == 1:
            active = active + 1
    return active


def average_tiredness(neurons, n_neurons):
    tiredness = 0
    for neuron in neurons:
        tiredness = tiredness + neuron.tiredness
    tiredness = tiredness / n_neurons
    return tiredness


def update_active_old(neurons):
    for neuron in neurons:
        neuron.active = neuron.active_old


def add_timestep_to_activity_plot(neurons, activity_per_neuron, step, number_of_neurons):
    for i in range(0, number_of_neurons -1):
        activity_per_neuron[i, step] = neurons[i].active_old


def calculate_entropy_approx(activity_per_neuron, step, number_of_neurons):
    #TODO: change network state in to token
    #TODO: calculate entropy based on token
    #TODO: calculate average entropy of each neuron...
    entr = 0
    for row in range(0,n_neurons):
        a = entropy(activity_per_neuron[row, :(step + 1)])
        entr = entr + a
    return entr/n_neurons
    #return np.apply_along_axis(entropy, 1, activity_per_neuron)


def calculate_hamming_distance_per_step(neurons, n_neurons):
    similar = 0
    for neuron in neurons:
        if neuron.active == neuron.active_old:
            similar = similar +1
    return (similar/n_neurons)


def simulation(max_connections, tiring_rate, number_of_neurons, simulation_steps, results, res_ind):
    neurons = []
    for i in range(0,number_of_neurons - 1):
        neurons.append(Neuron(max_connections, tiring_rate))

    for neuron in neurons:
        for i in range(0, max_connections):
            neuron.inputs.append(neurons[np.random.randint(0,number_of_neurons - 1)])
    activity_per_neuron = np.zeros((number_of_neurons, simulation_steps), dtype=np.bool)
    activity = []
    tired = []
    hamming_distances = []
    entropy_approx = []
    for i in range(0, simulation_steps):
        update_state(neurons)
        hamming_distances.append(calculate_hamming_distance_per_step(neurons, n_neurons))
        activity.append(active_neurons(neurons))
        tired.append(average_tiredness(neurons, number_of_neurons))
        update_active_old(neurons)
        add_timestep_to_activity_plot(neurons, activity_per_neuron, i, number_of_neurons)
        #entropy_approx.append(calculate_entropy_approx(activity_per_neuron, i, number_of_neurons))

    ax1 = plt.subplot(1,1,1)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Neuron#")
    plt.title("Neuron activity over time by neuron ID" + str(number_of_neurons) + " " + str(max_connections))
    plt.imshow(activity_per_neuron, cmap='Greys')
    '''ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Hamming distance")
    plt.title("Hamming distance in network from previous timestep")
    plt.plot(hamming_distances, color='black', lw=0.4)'''

    '''ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.ylabel("average shannon entropy in each neuron")
    plt.xlabel("Timestep (t)")
    plt.plot(entropy_approx, color='black', lw=0.4)'''
    '''ax4 = plt.subplot(4,1,4, sharex=ax1)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Shannon entropy")
    plt.title("Shannon entropy in network")
    plt.plot(entropy_approx, color='black', lw=0.4)'''

    plt.show()
    # results.loc[res_ind] = [max_connections, tiring_rate, number_of_neurons, simulation_steps, np.asarray(activity), np.asarray(tired)]


if __name__ == "__main__":
    simulation_steps = 500
    #tiring_rates = tqdm([0.1, 0.2, 0.3, 0.4, 0.5])
    tiring_rates = [0]
    #number_of_neurons = 100
    #max_connections = 3
    results = pd.DataFrame(columns=['max_connections', 'tiring_rate', 'neurons', 'simulation_steps', 'active_per_step', 'tired'])
    res_ind = 0
    for tiring_rate in tiring_rates:
        for n_neurons in tqdm(range(15, 16)):
            for max_connections in tqdm(range(2, 3)):
                for similar in range(0, 50):
                    simulation(max_connections, tiring_rate, n_neurons, simulation_steps, results, res_ind)
                    res_ind = res_ind+1
