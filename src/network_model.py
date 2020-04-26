import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


def sigmoid(z):
    """ Sigmoid function"""
    return (1.0 / (1.0 + np.exp(-z)))


class network(object):
    def __init__(self, sizes):  # sizes=[#inputs, #2nd layer, .... , #outputs]
        """ Initializing the network object
        sizes is array that will contain the number of neurons in each layer.
        the weights and biases for each neuron will be randomly chosen from a Norm~(meu,sigma) cdf"""
        print ('Initializing network model object...')
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        # Biases initiate - randomize by normal distribution biases as the numbers of neurons in the network (without
        # input neurons)
        sigma = 1
        meu = 0
        self.biases = [sigma * np.random.randn(dimensions, 1) + meu for dimensions in sizes[1:]]
        # dimensions: [#2nd layer, ... , #outputs], so we have X biases as the numbers of neurons except the inputs
        # Weights initiate - randomize by normal distribution weights for each neuron
        sigma2 = 1
        meu2 = 0
        self.weights = [sigma2 * np.random.randn(dim2, dim1) + meu2 for (dim1, dim2) in zip(sizes[:-1], sizes[1:])]
        # size[:-1], size[1:] creates pairs of elements in the sequence sizes with the next element,
        # because you pair up all elements except the last (sizes[:-1]) with all elements except the first (sizes[1:])

    def feed_forward(self, batch):
        """ Feeding forward the network with an intput of batch.
        That means we feed the network m times for an image which is a 784 input vector.
        :return the loss functions. 1. mean square error, 2. negative log loss"""
        # creating the function for each loss function
        out = [None] * len(batch)
        for i, (x, y) in enumerate(batch):
            x = np.reshape(x, (784, 1))
            for (b, w) in zip(self.biases, self.weights):
                x = sigmoid(np.dot(w, x) + b)  # a is now the output of the curr layer, and input of the next
            out[i] = np.reshape(x, len(batch))
        return [self.loss_mse(out, [i[1] for i in batch]),
                self.loss_nll(out, [i[1] for i in batch])]

    def loss_mse(self, out, y):
        """ :return the mean square error of @ out compared to @ y"""
        # y - is the labels, out is the network output
        mse = 0     # mean square error
        for i in range(0, len(y)):
            y_vec = np.zeros(10)
            y_vec[y[i]] = 1
            mse = mse + mean_squared_error(y_vec, out[i]) # calculating error from each label and output
        return mse/len(y) # for the mean square error fix

    def loss_nll(self, out, y):
        """ :return the negative log loss of @ out compared to @ y"""
        # y - is the labels, out is the network output
        nll = 0     # negative log los = -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        for i in range(0, len(y)):
            y_vec = np.zeros(10)
            y_vec[y[i]] = 1
            nll = nll + log_loss(y_vec, out[i]) # calculating error from each label and output
        return nll/len(y) # for the mean square error fix
