import numpy as np

class Neuron:
    def __init__(self, connections):
        self.connections = connections
        self.inputs = []
        self.state = 0
        self.next_state = 0
        self.input_from_other_neurons = 0
        self.firing_probability = 0
        self.base_firing_probability = 0.2
        self.tiring = 0
        self.tiring_map = [0.0,0.0,0.0,0.0,0.001, 0.01, 0.271, 0.8, 0.9 ]
        self.tiring_map_upper = len(self.tiring_map) - 1

    def tiring_modifier(self, tiring):
        return self.tiring_map[np.clip(tiring, 0, self.tiring_map_upper)]

    def input_to_probability_mapping(self, inp):
        return np.clip(inp/(self.connections) - self.tiring_modifier(self.tiring), 0, 1)

    def activation(self):
        for neuron in self.inputs:
            if neuron.state == 1:
                self.input_from_other_neurons += 1
        self.firing_probability = self.base_firing_probability + self.input_to_probability_mapping(self.input_from_other_neurons)
        if np.random.random() < self.firing_probability:
            self.next_state = 1
            self.tiring +=1
        else:
            self.next_state = 0
            self.tiring -=1

        self.input_from_other_neurons = 0

class Network:
    def __init__(self, seed, connections, n_neurons):
        self.neurons = []
        self.seed = seed
        self.connections = connections
        self.n_neurons = n_neurons
        np.random.seed(self.seed)

        # Initialize Neurons
        for i in range(0, n_neurons -1):
            self.neurons.append(Neuron(self.connections))

        # Initialize random state
        for i in range (0, n_neurons -1):
            self.neurons[i].state = np.random.randint(0,2)

        # Create Random Structure
        for neuron in self.neurons:
            for i in range(0,self.connections):
                neuron.inputs.append(self.neurons[np.random.randint(0, self.n_neurons -1)])

    def step(self):
        activity_per_neuron = np.zeros((self.n_neurons, 1), dtype=np.bool)
        for i in range(0, self.n_neurons - 1):
            self.neurons[i].activation()
            activity_per_neuron[i, 0] = self.neurons[i].state
            self.neurons[i].state = self.neurons[i].next_state
        return activity_per_neuron

    def add_stimuli(self, stimuli):
        for i in range(0, self.n_neurons -1):
            self.neurons[i].input_from_other_neurons += stimuli

    def simulate(self, steps, **kwargs):
        activity_per_neuron = np.zeros((self.n_neurons, steps), dtype=np.bool)
        hamming_distance = []
        for step in range(0, steps):
            # Hamming distance
            hamm = 0
            for i in range(0,self.n_neurons - 1):
                # Activate
                self.neurons[i].activation()
                # Read State
                activity_per_neuron[i, step] = self.neurons[i].state
                # Calculate Hamming Distance
                if i !=0:
                    if self.neurons[i].next_state == self.neurons[i].state:
                        hamm += 1
                self.neurons[i].state = self.neurons[i].next_state
            # add hamming distance
            hamming_distance.append(hamm)
        
        return activity_per_neuron, hamming_distance