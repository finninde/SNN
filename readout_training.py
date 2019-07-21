from network import Network, Neuron
from readout_layer import ReadoutLayer

if __name__=="__main__":

    for i in range(0,1):
        simulation_steps = 4
        connections = 3
        seed = 3 + 40*i
        n_neurons = 5
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
            network.add_stimuli(parity['train']['inp'][i])
            activity.append(network.step())

        '''Train everything simultaneous'''
        print(len(activity[-len(parity['train']['inp']):]))
        Xs, ys = activity[-len(parity['train']['inp']):], parity['train']['out']
        readout.train(activity[-len(parity['train']['inp']):],
                      parity['train']['out'])

        '''Run network for some more time so to reset'''
        for steps in range(0, 30):
            activity.append(network.step())

        positives = 0
        negatives = 0
        for steps in range(0, len(parity['evaluate']['inp'])):
            network.add_stimuli(parity['evaluate']['inp'][i])
            activity.append(network.step())
            if readout.predict(activity[-1]) == parity['evaluate']['out'][i]:
                positives += 1
            else:
                negatives += 1

        print("---------------------")
        print("positives: " + str(positives))
        print("negatives: " + str(negatives))
        print("accuracy: " + str(positives/(positives + negatives)))
        print("Configuration: ")
        print("Connections" + str(connections))
        print("n_neurons"+ str(n_neurons))
        print("task: PARITY" )