from network import Network, Neuron
from readout_layer import ReadoutLayer
import pickle
from tqdm import tqdm
from training_test_generator import *

if __name__=="__main__":
    parity = {}
    X_train, y_train = generate_training_set(20, 20, 20, 60, 81549300)
    parity['train'] = {}
    parity['train']['inp'] = [0]*40
    parity['train']['out'] = y_train

    X_eval, y_eval = generate_evaluation_set(10, 10, 10, 15, 69696969)
    parity['evaluate'] = {}
    parity['evaluate']['inp'] = [1]*40
    parity['evaluate']['out'] = y_eval

    sep_inp = []
    tot_sep_inp = []

    for nevroner in tqdm(range(2, 100)):
        sep_inp.append([])
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
                network.add_stimuli(1)
                activity.append(network.step())

            # Network 2
            activity_n_2 = []
            simulation_steps = 4
            connections = 3
            seed = 3 + 40 * i
            n_neurons = nevroner
            network = Network(seed, connections, n_neurons)
            readout = ReadoutLayer(layer_type="linear", dim=20)
            '''Run network for initial steps'''
            for steps in range(0, 100):
                activity_n_2.append(network.step())

            '''Add stimuli and train'''
            for steps in range(0, len(parity['evaluate']['inp'])):
                network.add_stimuli(0)
                activity_n_2.append(network.step())

            # Calculate distances
            #Remember, not to measure initial settling phase

            cumulative_distance = 0
            for timestep in range(100, 139):
                sum_of_similar_neurons = 0
                for neuron in range(0, n_neurons):
                    if activity[timestep][neuron] == activity_n_2[timestep][neuron]:
                        sum_of_similar_neurons +=1
                distance = sum_of_similar_neurons/n_neurons
                cumulative_distance += distance
                sep_inp[-1].append(distance)

        tot_sep_inp.append(cumulative_distance/40)

        with open('results/sep_inp_n_neurons/all_sep_for_boxplot_' + str(nevroner) + '.pckl', 'wb')as f:
            pickle.dump(sep_inp, f)
        f.close()
        with open('results/sep_inp_n_neurons/average_for_each_neuron' + str(nevroner) + '.pckl', 'wb') as g:
            pickle.dump(tot_sep_inp, g)
        g.close()