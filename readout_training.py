from network import Network, Neuron
from readout_layer import ReadoutLayer
import pickle
from tqdm import tqdm


if __name__=="__main__":
    totacc = []
    for nevroner in tqdm(range(5, 20)):
        accuracies = []
        for i in range(0,900):
            simulation_steps = 4
            connections = 3
            seed = 3 + 40*i
            n_neurons = nevroner
            activity = []
            network = Network(seed, connections, n_neurons)
            readout = ReadoutLayer(layer_type="linear", dim=20)
            parity = {}
            parity['train'] = {}
            parity['train']['inp'] = [1,0,1,1,0,1,1,1,1,0]
            parity['train']['out'] = [0,0,0,1,0,0,1,0,1,0]

            parity['evaluate'] = {}
            parity['evaluate']['inp'] = [1,0,1,1,0,1,1,1,1,0]
            parity['evaluate']['out'] = [0,0,0,1,0,0,1,0,1,0]

            '''Run network for initial steps'''
            for steps in range(0,100):
                activity.append(network.step())

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

        print("---------------")
        print("n_nerons: " + str(nevroner))
        print("total acc: " + str(sum(accuracies) / len(accuracies)))
        totacc.append(accuracies)
    with open('totacc.pckl', 'wb')as f:
        pickle.dump(totacc, f)
    f.close()