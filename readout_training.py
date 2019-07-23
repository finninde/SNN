from network import Network, Neuron
from readout_layer import ReadoutLayer
import pickle
from tqdm import tqdm
from training_test_generator import *

if __name__=="__main__":
    a_acc = []
    totacc = []
    # Use same training and test set for all of them, but use a large one

    parity = {}
    X_train, y_train = generate_training_set(20, 20, 20, 60, 81549300)
    parity['train'] = {}
    parity['train']['inp'] = X_train
    parity['train']['out'] = y_train

    X_eval, y_eval = generate_evaluation_set(10, 10, 10, 15, 69696969)
    parity['evaluate'] = {}
    parity['evaluate']['inp'] = X_eval
    parity['evaluate']['out'] = y_eval

    for nevroner in tqdm(range(2, 343)):
        accuracies = []
        for i in range(0,900):
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
                print(activity)

            '''Add stimuli and train'''
            for steps in range(0, len(parity['train']['inp'])):
                network.add_stimuli(parity['train']['inp'][steps])
                activity.append(network.step())

            '''Train everything simultaneous'''
            #print(len(activity[-len(parity['train']['inp']):]))
            Xs = activity[-len(parity['train']['inp']):]

            readout.train(Xs,
                          parity['train']['out'])

            '''Run network for some more time so to reset'''
            for steps in range(0, 30):
                activity.append(network.step())

            positives = 0
            negatives = 0
            for steps in range(0, len(parity['evaluate']['inp'])):
                network.add_stimuli(parity['evaluate']['inp'][steps])
                activity.append(network.step())


            readout_out = readout.predict(activity[-len(parity['evaluate']['inp']):])
            print(readout_out)
            for output_of_RC in range (0, len(parity['evaluate']['inp'])):
                if readout_out[output_of_RC] > 0.5:
                    readout_out[output_of_RC] = 1
                else:
                    readout_out[output_of_RC] = 0

            for output_of_RC in range(0, len(parity['evaluate']['out'])):
                if readout_out[output_of_RC] == parity['evaluate']['out'][output_of_RC]:
                    positives += 1
                else:
                    negatives += 1


            #print("---------------------")
            #print("positives: " + str(positives))
            #print("negatives: " + str(negatives))
            #print("accuracy: " + str(positives/(positives + negatives)))
            #print("Configuration: ")
            #print("Connections: " + str(connections))
            #print("n_neurons: "+ str(n_neurons))
            #print("task: PARITY" )
            accuracies.append((positives/(positives + negatives)))

        #print("---------------")
        #print("n_nerons: " + str(nevroner))
        #print("total acc: " + str(sum(accuracies) / len(accuracies)))
        a_acc.append(sum(accuracies)/len(accuracies))
        totacc.append(accuracies)
        with open('results/n_neurons_real_training_test/all_accuracies_for_boxplot_' + str(nevroner) + '.pckl', 'wb')as f:
            pickle.dump(totacc, f)
        f.close()
        with open('results/n_neurons_real_training_test/average_accuracy_n_neuron_' + str(nevroner) + '.pckl', 'wb') as g:
            pickle.dump(a_acc, g)
        g.close()