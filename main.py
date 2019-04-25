import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

class Neuron():
    def __init__(self, max_connections, tiring_rate):
        self.outputs = [] 
        self.inputs = []
        self.inp = 0
        self.firing_probability = 0
        self.tiredness = 0
        # Juster hvor mye av nettet jeg starter med å aktivere
        self.active_prev = np.random.randint(0,2)
        self.active = np.random.randint(0,2)
        self.first_run = True
        self.max_connections = max_connections
        self.tiring_rate = tiring_rate

    def update(self):
        if self.first_run == True:
            if self.active:
                self.tiredness = np.clip(self.tiredness + self.tiring_rate, 0, 1)
                self.first_run = False
            else:
                self.active  = 0
                self.tiredness = np.clip(self.tiredness - self.tiring_rate, 0, 1)
                self.first_run = False          
        else:
            self.threshold()
            if self.active:
                self.tiredness = np.clip(self.tiredness - self.tiring_rate, 0, 1)
            else:
                self.tiredness =  np.clip(self.tiredness + self.tiring_rate, 0, 1)
        
    def threshold(self):
        for neuron in self.inputs:
            if neuron.active:
                self.inp = self.inp + 1
        if self.inp == 0:
            self.firing_probability = 0
        else:
            self.firing_probability = np.clip(0.5 + (self.inp/2*(self.max_connections+1)) - self.tiredness, 0, 1)
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
            
def simulation(max_connections, tiring_rate, number_of_neurons, simulation_steps, results, res_ind):
    neurons = []

    for i in range(0,number_of_neurons):
        neurons.append(Neuron(max_connections, tiring_rate))

    for neuron in neurons:
        for i in range(0, np.random.randint(0,max_connections)):
            neuron.inputs.append(neurons[np.random.randint(0,number_of_neurons)])
    
    activity = []
    for i in range(0, simulation_steps):
        update_state(neurons)
        activity.append(active_neurons(neurons))

    results.loc[res_ind] = [max_connections, tiring_rate, number_of_neurons, simulation_steps, np.asarray(activity)]
    

if __name__=="__main__":
    simulation_steps = 20
    tiring_rates = tqdm([0.1, 0.2, 0.3, 0.4, 0.5])
    number_of_neurons = 10
    max_connections = 3
    results = pd.DataFrame(columns=['max_connections', 'tiring_rate', 'neurons', 'simulation_steps', 'active_per_step'])
    res_ind = 0
    '''for i in range(0,100):
        simulation(4,0.1,10,100,results,  i)'''
    for tiring_rate in tiring_rates:
        for number_of_neurons in tqdm(range(3,100)):
            for max_connections in range(1,50):
                simulation(max_connections, tiring_rate, number_of_neurons, simulation_steps, results, res_ind)
                res_ind = res_ind+1
    results.to_pickle('res.pckl')
    