import math
import random
import numpy as np
MAX_CONNECTIONS = 10
TIRING_RATE = 0.1
NUMBER_OF_NEURONS = 40
SIMULATION_STEPS = 100

class Neuron():
    def __init__(self):
        self.outputs = [] 
        self.inputs = []
        self.inp = 0
        self.firing_probability = 0
        self.tiredness = 0
        # Juster hvor mye av nettet jeg starter med å aktivere
        self.active = np.random.randint(0,2)
        self.first_run = True

    def update(self):
        if self.first_run == True:
            if self.active:
                self.tiredness = np.clip(self.tiredness + TIRING_RATE, 0, 1)
                self.first_run = False
            else:
                self.active  = 0
                self.tiredness = np.clip(self.tiredness - TIRING_RATE, 0, 1)
                self.first_run = False          
        else:
            self.threshold()
            if self.active:
                self.tiredness = np.clip(self.tiredness - TIRING_RATE, 0, 1)
            else:
                self.tiredness =  np.clip(self.tiredness + TIRING_RATE, 0, 1)
        
    def threshold(self):
        for neuron in self.inputs:
            if neuron.active:
                self.inp = self.inp + 1
        if self.inp == 0:
            self.firing_probability = 0
        else:
            self.firing_probability = np.clip(0.5 + (self.inp/2*(MAX_CONNECTIONS+1)) - self.tiredness, 0, 1)
        self.inp = 0
        if random.random() < self.firing_probability:
            self.active = 1
        else:
            self.active = 0        

def update_state(neurons):
    for neuron in neurons:
        neuron.update()

def active_neurons(neurons):
    active = 0
    for neuron in neurons:
        if neuron.active == 1:
            active = active + 1
    return active
            

if __name__=="__main__":
    neurons = []

    for i in range(0,NUMBER_OF_NEURONS):
        neurons.append(Neuron())

    for neuron in neurons:
        for i in range(0, np.random.randint(0,MAX_CONNECTIONS)):
            neuron.inputs.append(neurons[np.random.randint(0,NUMBER_OF_NEURONS)])
    
    activity = []
    for i in range(0, SIMULATION_STEPS):
        update_state(neurons)
        activity.append(active_neurons(neurons))

    print(activity)
