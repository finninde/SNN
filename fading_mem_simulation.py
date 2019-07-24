from network import Network, Neuron
from readout_layer import ReadoutLayer
import pickle
from tqdm import tqdm
from training_test_generator import *

if __name__=="__main__":
    parity = {}
    X_train, y_train = generate_training_set(20, 20, 20, 60, 81549300)
    parity['train'] = {}
    parity['train']['inp'] = X_train
    parity['train']['out'] = y_train

    X_eval, y_eval = generate_evaluation_set(10, 10, 10, 15, 69696969)
    parity['evaluate'] = {}
    parity['evaluate']['inp'] = X_eval
    parity['evaluate']['out'] = y_eval
    fading_memory = []
    tot_fading_mem = []
    for nevroner in tqdm(range(2, 100)):
        fading_memory.append([])
        for i in range(0,900):
            # Network 1
            simulation_steps = 4
            connections = 3
            seed = 3 + 40*i
            n_neurons = nevroner
            activity = []
            network = Network(seed, connections, n_neurons)
            readout = ReadoutLayer(layer_type="linear", dim=20)
            '''Run network for initial steps'''
            for steps in range(0,100):
                activity.append(network.step())

            '''Add stimuli and train'''
            for steps in range(0, len(parity['train']['inp'])):
                network.add_stimuli(parity['train']['inp'][steps])
                activity.append(network.step())
            # Network 2
            activity_n_2 = []
            simulation_steps = 4
            connections = 3
            seed = (3 + 40 * i)*2
            n_neurons = nevroner
            network = Network(seed, connections, n_neurons)
            readout = ReadoutLayer(layer_type="linear", dim=20)
            '''Run network for initial steps'''
            for steps in range(0, 100):
                activity_n_2.append(network.step())

            '''Add stimuli and train'''
            for steps in range(0, 39):
                network.add_stimuli(parity['train']['inp'][steps])
                activity_n_2.append(network.step())
            
            #Remember, not to measure initial settling phase
            cumulative_distance = 0
            for timestep in range(100, 139):
                sum_of_similar_neurons = 0
                for neuron in range(0, n_neurons):
                    if activity[timestep][neuron] == activity_n_2[timestep][neuron]:
                        sum_of_similar_neurons +=1
                distance = sum_of_similar_neurons/n_neurons
                cumulative_distance += distance
                fading_memory[-1].append(distance)
        tot_fading_mem.append(cumulative_distance/40)

        with open('results/fading_mem_n_neurons/all_fadings_for_boxplot_' + str(nevroner) + '.pckl', 'wb')as f:
            pickle.dump(fading_memory, f)
        f.close()
        with open('results/fading_mem_n_neurons/average_for_each_neuron' + str(nevroner) + '.pckl', 'wb') as g:
            pickle.dump(tot_fading_mem, g)
        g.close()