import time
import mnist_dreader
import numpy as np
import operator
import random as rnd
import network_model


class SGD(object):
    def __init__(self, network, eta, batch_size, epochs_size):
        """ Initializing the SGD optimizer object:
        @network is the neural network,
        @eta is the learing rate
        @batch_size is the batch size
        @epochs_size is the number of epochs that will be running"""
        self.network = network
        self.eta = eta
        self.batch_size = batch_size
        self.epochs_size = epochs_size
        self.data_reader = mnist_dreader.dataReader()  # constructing mnist data reader object
        self.total_batches = np.floor(len(self.data_reader.train) / batch_size)

    def step(self):
        """ 1 step for 1 batch - updates the network's weights and biases
        :returns negative log loss, gradient norms"""
        batch = self.data_reader.get_batch(self.batch_size, 'train')
        # calculating negative log loss
        #[_, _, neg_log_loss] = self.feed_forward(batch, 0) # TODO: uncomment
        neg_log_loss = 0    # TODO: DELETE
        # getting gradient of weights and biases
        nabla_biases, nabla_weights = self.network.back_prop(batch)
        # updating them
        self.network.update_wb(nabla_biases, nabla_weights, self.eta)
        return neg_log_loss, np.abs(nabla_biases), np.abs(nabla_weights)  # saving training: NLL, gradient norms

    def train_epoch(self):
        """ Training on all the training sets
        :returns negative log loss, avg/min/max gradient norm"""
        self.data_reader.reset_batch_num()  # start from the first batch
        self.data_reader.shuffle_train()  # shuffle the training examples
        neg_log_loss = 0  # saving the negative log loss of the training
        gradient_norms = []  # array for gradient norms
        for i in range(int(self.total_batches)):
            nll_delta, nb, nw = self.step()  # 1 batch, updates the weights and biases
            neg_log_loss += nll_delta

            # TODO: find a way to get max avg and min of the weights and biases FAST
            # for b_layer in range(len(self.network.biases)):
            #     for b_neuron in self.network.biases[b_layer]:
            #         gradient_norms.append(b_neuron[0])
            # for w_layer in range(len(self.network.weights)):
            #     for w_neuron in range(len(self.network.weights[w_layer])):
            #         for w_in in self.network.weights[w_layer][w_neuron]:
            #             gradient_norms.append(w_in)

        nll = neg_log_loss / self.total_batches
        max_gn = 0  # np.amax(gradient_norms)      # max gradient norm #TODO: FIX IT
        min_gn = 0  # np.amin(gradient_norms)      # min gradient norm
        avg_gn = 0  # np.mean(gradient_norms)      # average gradient norm

        return nll, avg_gn, min_gn, max_gn  # return negative log loss of the training

    def predict(self, test_data):
        """ Returns the number of the inputs for which the network
        outputs the correct result.
        result is by the maximum activation"""
        test_data_out = self.network.feed_forward(test_data, 1)
        data_result = [(np.argmax(a), y[1])
                       for (a, y) in zip(test_data_out, test_data)]
        return sum(int(x == y) for (x, y) in data_result)

    def training_program(self):
        """ Training  the network for epochs_size epochs """
        for i in range(self.epochs_size):
            print ("Started Epoch {0}...").format(i)
            start_time = time.time()  # counting time
            nll, avg_gn, min_gn, max_gn = self.train_epoch()  # training 1 epoch + saving NLL
            sum_train = self.predict(self.data_reader.train)  # checking the valid trains (55k)
            max_train = len(self.data_reader.train)
            sum_valid = self.predict(self.data_reader.valid)  # checking the valid tests (5k)
            max_valid = len(self.data_reader.valid)
            elapsed_time = time.time() - start_time
            print ("Epoch {0}:\n"
                   "    Negative Log Loss: {1}\n"
                   "    Accuracy training examples: {2} / {3}\n"
                   "    Accuracy validation examples: {4} / {5}\n"
                   "    Gradient norm: Average = {6}, Max = {7}, Min = {8}\n"
                   "    Epoch elapsed time: {9} seconds."). \
                format(i, nll, sum_train, max_train, sum_valid, max_valid, avg_gn, max_gn, min_gn, elapsed_time)
