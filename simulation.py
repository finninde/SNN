from network import Network, Neuron
import matplotlib.pyplot as plt

if __name__ == "__main__":
    simulation_steps = 500
    connections = 3
    seed = 81549300
    n_neurons = 20
    network = Network(seed, connections, n_neurons)
    activity_per_neuron, hamming_distance = network.simulate(simulation_steps)
    ax1 = plt.subplot(2,1,1)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Neuron#")
    plt.title("Neuron activity over time by neuron ID" )
    plt.imshow(activity_per_neuron, cmap='Greys')
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    plt.xlabel("Timestep (t)")
    plt.ylabel("Hamming distance")
    plt.title("Hamming distance in network from previous timestep")
    plt.plot(hamming_distance, color='black', lw=0.4)
    plt.show()