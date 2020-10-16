import numpy as np
import tensorflow.keras.datasets.mnist as mnist
import random as rnd
import time

""" seed = 30

rnd.seed(30)
np.random.seed(30) """

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))

class Network:
    """Neural network that takes in an array containing
    amount of neurons in each layer including input and output"""
    def __init__(self, arch):
        self.num_layers = len(arch)
        self.arch = arch
        self.weights = [ np.random.randn(y, x) for y, x in zip(arch[1:], arch[:-1]) ]
        self.biases = [ np.random.normal(0, 1, y) for y in arch[1:] ]

    def feedforward(self, x):
        """Returns the output vector (i.e. the networks choice)
        for a given input vector x"""
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(w @ x + b)
        return x

    def back_prop(self, x, y):
        """ Computes the gradient of all the biases
        and weights in the network for a given input/output pair """
        # These all togehther make up the gradient of the network for
        # a given input x.
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]

        # Lists of z and a for each layers
        weighted_sums = []
        activations = [x]

        # Compute activations and weighted sums for each layer
        a = x
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            weighted_sums.append(z)
            a = sigmoid(z)
            activations.append(a)
        
        # Error for the final layer
        delta_l = (activations[-1] - y) * sigmoid_prime(weighted_sums[-1])
        nabla_b[-1] = delta_l
        nabla_w[-1] = np.outer(delta_l, activations[-2])

        # Compute the delta_w's and delta_b's and rest of errors
        for l in range(2, self.num_layers):
            z_l = weighted_sums[-l]
            # Compute the error for next layer
            w_T = self.weights[-l+1].transpose()
            sp = sigmoid_prime(z_l)
            mul = w_T @ delta_l
            delta_l = mul * sp
            nabla_b[-l] = delta_l
            nabla_w[-l] = np.outer(delta_l, activations[-l - 1])

        return (nabla_w, nabla_b)

    def sgd(self, train_data, epochs, eta, mini_batch_size, test_data=None):
        """Trains the network"""
        # shuffle data
        for i in range(epochs):
            t0 = time.time()

            rnd.shuffle(train_data)
            for j in range(0, len(train_data), mini_batch_size):
                # Update network for each minibatch
                mini_batch = train_data[j:j+mini_batch_size]
                self.update_network(mini_batch, eta)

            # Evaluate the network
            if test_data != None:

                num_correct = 0
                for x, y in test_data:
                    if y[ np.argmax(self.feedforward(x)) ] == 1:
                        num_correct += 1
                accuracy = 100 * num_correct / len(test_data)

                print(f"Epoch {i+1}: Classification accuracy {accuracy:.2f}%")

            time_elapsed = time.time() - t0
            print(f"Elapsed time: {time_elapsed:.2f} s")




    def update_network(self, mini_batch, eta):
        m = len(mini_batch)
        nabla_w = [ np.zeros(w.shape) for w in self.weights ]
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.back_prop(x, y)
            nabla_w = [ nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w) ]
            nabla_b = [ nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b) ]

        self.weights = [ w - (eta / m)*nw
                        for w, nw in zip(self.weights, nabla_w) ]
        self.biases = [ b - (eta / m)*nb
                        for b, nb in zip(self.biases, nabla_b) ]
            




    

def main():
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 28*28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28*28))
    x_train = x_train / 255
    x_test = x_test / 255

    arch = [28*28, 38, 10]
    network = Network(arch)

    y_train = [ np.zeros(10) for l in labels_train ]
    for i in range(len(labels_train)):
        y_train[i][labels_train[i]] = 1

    y_test = [ np.zeros(10) for l in labels_test ]
    for i in range(len(labels_test)):
        y_test[i][labels_test[i]] = 1

    network.sgd(
        train_data=list(zip(x_train, y_train)),
        epochs=100,
        eta=0.5,
        mini_batch_size=10,
        test_data=list(zip(x_test, y_test)),
    )

if __name__ == "__main__":
    main()