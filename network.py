import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

class Network:
    """Neural network that takes in an array containing
    amount of neurons in each layer including input and output"""
    def __init__(self, arch):
        self.num_layers = len(arch)
        self.arch = arch
        self.weights = [ np.random.rand(y, x) for y, x in zip(arch[1:], arch[:-1]) ]
        self.biases = [ np.random.normal(0, 1, y) for y in arch[1:] ]

    def predict(self, x):
        """Returns the output vector (i.e. the networks choice)
        for a given input vector x"""
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(w @ x + b)
        return x

    def back_prop(self, x, y):
        """ Computes the gradient of all the biases
        and weights in the network for a given input/output pair """
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        weighted_sums = []
        activations = [x]

        # Compute activations and weighted sums for each layer
        a = x
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            weighted_sums.append(z)
            a = sigmoid(z)
            activations.append(a)
        
        delta_l = (activations[-1] - y) * sigmoid_prime(weighted_sums[-1])

        # TODO: compute more


        return (nabla_w, nabla_b)

    def sgd(self, train_data, epochs, eta):
        pass

    def update_network(self, mini_batch):
        pass



    

def main():
    arch = [3, 5, 4]
    network = Network(arch)
    print(network.predict(np.random.normal(0, 1, arch[0])))

if __name__ == "__main__":
    main()