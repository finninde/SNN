from network import Network, Neuron
from readout_layer import ReadoutLayer
import pickle
from tqdm import tqdm
from training_test_generator import *
import random

if __name__=="__main__":
    a_acc = []
    totacc = []
    # Use same training and test set for all of them, but use a large one

    parity = {}
    X_train, y_train = generate_training_set(90, 90, 90, 30, 81549300)
    parity['train'] = {}
    parity['train']['inp'] = X_train
    parity['train']['out'] = y_train

    X_eval, y_eval = generate_evaluation_set(10, 90, 10, 700, 69696969)
    parity['evaluate'] = {}
    parity['evaluate']['inp'] = X_eval
    parity['evaluate']['out'] = y_eval

    for nevroner in tqdm(range(10,200)):
        accuracies = []
        for i in range(0,900):
            simulation_steps = 4
            connections = 3
            seed = 3 + 40*i
            activity = []
            #print(n_neurons)
            network = Network(seed, connections, nevroner)
            readout = ReadoutLayer(layer_type="perceptron", dim=20)
            #print(readout)
            #TODO: generer en liste med nevroner du vil hente out fra
            # Append kun de nevronene til activity
            # Ferdig
            visible = 0.6

            visible_neurons = np.around(visible * nevroner)
            #TODO: Generate an index list of which neurons to pick
            index_list = random.sample(range(0, nevroner), int(visible_neurons))

            '''Run network for initial steps'''
            for steps in range(0,100):
                temp = network.step()
                templist = np.zeros((int(visible_neurons)), dtype=np.bool)
                k = 0
                for random_element in index_list:
                    templist[k] = temp[random_element]
                    k += 1
                activity.append(templist)

            '''Add stimuli and train'''
            for steps in range(0, len(parity['train']['inp'])):
                network.add_stimuli(parity['train']['inp'][steps])
                temp = network.step()
                templist = np.zeros((int(visible_neurons)), dtype=np.bool)
                k = 0
                for random_element in index_list:
                    templist[k] = temp[random_element]
                    k += 1
                activity.append(templist)

            '''Train everything simultaneous'''
            #print(len(activity[-len(parity['train']['inp']):]))
            Xs = activity[-len(parity['train']['inp']):]

            readout.train(Xs,
                          parity['train']['out'])

            '''Run network for some more time so to reset'''
            for steps in range(0, 30):
                temp = network.step()
                templist = np.zeros((int(visible_neurons)), dtype=np.bool)
                k = 0
                for random_element in index_list:
                    templist[k] = temp[random_element]
                    k += 1
                activity.append(templist)

            positives = 0
            negatives = 0


            for steps in range(0, len(parity['evaluate']['inp'])):
                network.add_stimuli(parity['evaluate']['inp'][steps])
                temp = network.step()
                templist = np.zeros((int(visible_neurons)), dtype=np.bool)
                k = 0
                for random_element in index_list:
                    templist[k] = temp[random_element]
                    k += 1
                activity.append(templist)

            # TODO: Readout trenger å predicte baser på activity
            #abc = activity[-len(parity['evaluate']['inp'])]
            #abd = parity['evaluate']['out']
            #readout_out = readout.predict(activity[-len(parity['evaluate']['inp']):])
            #print(readout_out)
            '''for output_of_RC in range (0, len(parity['evaluate']['inp'])):
                if readout_out[output_of_RC] > 0.5:
                    readout_out[output_of_RC] = 1
                else:
                    readout_out[output_of_RC] = 0'''
            #print(readout.reg.predict(activity[-len(parity['evaluate']['inp']):]))
            #print(parity['evaluate']['out'])
            accuracies.append(readout.reg.score(activity[-len(parity['evaluate']['inp']):],
                                    parity['evaluate']['out']))

            #for output_of_RC in range(0, len(parity['evaluate']['out'])):
            #    if readout_out[output_of_RC] == parity['evaluate']['out'][output_of_RC]:
            #        positives += 1
            #    else:
            #        negatives += 1

            #print("---------------------")
            #print("positives: " + str(positives))
            #print("negatives: " + str(negatives))
            #print("accuracy: " + str(positives/(positives + negatives)))
            #print("Configuration: ")
            #print("Connections: " + str(connections))
            #print("n_neurons: "+ str(n_neurons))
            #print("task: PARITY" )
            #accuracies.append((positives/(positives + negatives)))

        #print("---------------")
        #print("n_nerons: " + str(nevroner))
        #print("total acc: " + str(sum(accuracies) / len(accuracies)))
        a_acc.append(sum(accuracies)/len(accuracies))
        totacc.append(accuracies)
        with open('results/n_neurons_real_training_test_visibility/all_accuracies_for_boxplot_' + str(nevroner) + '.pckl', 'wb')as f:
            pickle.dump(totacc, f)
        f.close()
        with open('results/n_neurons_real_training_test_visibility/average_accuracy_n_neuron_' + str(nevroner) + '.pckl', 'wb') as g:
            pickle.dump(a_acc, g)
        g.close()