import time
import mnist_dreader
import numpy as np
import matplotlib.pyplot as plt
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

    def step(self, optimizer):
        """ 1 step for 1 batch - updates the network's weights and biases
        :returns negative log loss, gradient norms"""
        batch = self.data_reader.get_batch(self.batch_size, 'train')
        len_training_set = len(self.data_reader.train)

        if optimizer == 'SGD':             # SGD is the normal gradient descent
            # getting gradient of weights and biases
            nabla_biases, nabla_weights = self.network.back_prop(batch)
            # updating them
            self.network.update_wb(nabla_biases, nabla_weights, self.eta, len(batch), len_training_set)
        elif optimizer == 'Momentum':      # Momentum optimizer- check in here -> https://ruder.io/optimizing-gradient-descent/index.html#rmsprop
            # getting gradient of weights and biases
            nabla_biases, nabla_weights = self.network.back_prop(batch)
            # updating them
            self.network.update_wb_momentum(nabla_biases, nabla_weights, self.eta, len(batch))
        elif optimizer == 'RMSProp':         # Adam optimizer
            # getting gradient of weights and biases
            nabla_biases, nabla_weights, nabla_sqr_b, nabla_sqr_w = self.network.back_prop_RMSprop(batch)
            # updating them
            self.network.update_wb_RMSprop(nabla_biases, nabla_weights, nabla_sqr_b, nabla_sqr_w, self.eta, len(batch))

        return np.abs(nabla_biases), np.abs(nabla_weights)  # saving training: NLL, gradient norms

    def train_epoch(self, optimizer):
        """ Training on all the training sets
        :returns negative log loss, avg/min/max gradient norm"""
        self.data_reader.reset_batch_num()  # start from the first batch
        self.data_reader.shuffle_train()  # shuffle the training examples
        gradient_norms = []  # array for gradient norms
        for i in range(int(self.total_batches)):
            nb, nw = self.step(optimizer)  # 1 batch, updates the weights and biases

            # TODO: find a way to get max avg and min of the weights and biases FAST - comment it to fast things up
            # for b_layer in range(len(nb)):
            #     for b_neuron in nb[b_layer]:
            #         gradient_norms.append(b_neuron[0])
            #         for w_in in nw[w_layer][w_neuron]:
            #             gradient_norms.append(w_in)
            ## for w_layer in range(len(nw)):
            ##     for w_neuron in range(len(nw[w_layer])):
            ##         for w_in in nw[w_layer][w_neuron]:
            ##             gradient_norms.append(w_in)

        max_gn = 0 #np.amax(gradient_norms)      # max gradient norm #TODO: comment to run fast
        min_gn = 0 #np.amin(gradient_norms)      # min gradient norm
        avg_gn = 0 #np.mean(gradient_norms)      # average gradient norm

        return avg_gn, min_gn, max_gn  # return negative log loss of the training

    def predict(self, test_data, loss):
        """ Returns the number of the inputs for which the network
        outputs the correct result.
        result is by the maximum activation"""
        if loss == 'loss':
            test_data_out, loss_mse, loss_nll = self.network.feed_forward(test_data, 0)  # feed forward for a batch
            data_result = [(np.argmax(out), tup_xy[1])  # argmax from result and y-s
                           for (out, tup_xy) in zip(test_data_out, test_data)]
            return sum(int(x == y) for (x, y) in data_result), loss_nll
        else:
            test_data_out = self.network.feed_forward(test_data, 1)     # feed forward for a batch
            data_result = [(np.argmax(out), tup_xy[1])                # argmax from result and y-s
                           for (out, tup_xy) in zip(test_data_out, test_data)]
            return sum(int(x == y) for (x, y) in data_result)

    def training_program(self, optimizer):
        """ Training  the network for epochs_size epochs """
        nll_learn_curve = []
        train_passed = []
        valid_passed = []
        max_gradient = []
        min_gradient = []
        avg_gradient = []
        for i in range(self.epochs_size):
            print ("Started Epoch {0}...").format(i)
            start_time = time.time()  # counting time
            avg_gn, min_gn, max_gn = self.train_epoch(optimizer)  # training 1 epoch + saving NLL
            sum_train = self.predict(self.data_reader.train, '')  # checking the valid trains (55k)
            max_train = len(self.data_reader.train)
            sum_valid, nll = self.predict(self.data_reader.valid, 'loss')  # checking the valid tests (5k)
            max_valid = len(self.data_reader.valid)
            elapsed_time = time.time() - start_time
            print ("EPOCH {0} - for {10} optimizer with eta {16}, {13} cost function, regularization {14} with lambda {15}:\n"
                   "    Negative Log Loss (Valid): {1}\n"
                   "    Accuracy training examples: {2} / {3}  =  {11}%\n"
                   "    Accuracy validation examples: {4} / {5}  =  {12}%\n"
                   #"    Gradient norm: Average = {6}, Max = {7}, Min = {8}\n"
                   "    Epoch elapsed time: {9} seconds."). \
                format(i, round(nll, 4), sum_train, max_train, sum_valid, max_valid, avg_gn, max_gn, min_gn, elapsed_time,
                       optimizer, round(100*sum_train/(1.0*max_train), 2), round(100*sum_valid/(1.0*max_valid), 2),
                       self.network.cost, self.network.regularization, self.network.reg_lambda, self.eta)
            nll_learn_curve.append(nll)
            train_passed.append(sum_train)
            valid_passed.append(sum_valid)
            max_gradient.append(max_gn)
            min_gradient.append(min_gn)
            avg_gradient.append(avg_gn)

        epoch = range(self.epochs_size)
        # plotting learning curve (nll w.r.t. epoch)
        plt.plot(epoch, nll_learn_curve)
        plt.xlabel('Epoch #')
        plt.ylabel('Negative Log Loss (w.r.t epoch)')
        plt.title('Learning Curve - Negative log loss cost\n{} optimizer with eta {}, {} cost function\nregularization {} with lambda {}'
                  .format(optimizer, self.network.cost, self.eta, self.network.regularization, self.network.reg_lambda))
        plt.grid()
        plt.show()

        # plotting accuracy curve
        plt.subplot(211)
        plt.plot(epoch, train_passed)
        plt.xlabel('Epoch #')
        plt.ylabel('Training passed')
        plt.grid()
        plt.title('Training 55k and Validation 5k Accuracy Curve\n{} optimizer with eta {}, {} cost function\nregularization {} with lambda {}'
                  .format(optimizer, self.network.cost, self.eta, self.network.regularization, self.network.reg_lambda))
        plt.subplot(212)
        plt.plot(epoch, valid_passed)
        plt.xlabel('Epoch #')
        plt.ylabel('Validations passed')
        plt.grid()
        plt.show()

        # plotting gradient norms
        plt.plot(epoch, min_gradient, 'b', label='min')
        plt.plot(epoch, avg_gradient, 'g', label='avg')
        plt.plot(epoch, max_gradient, 'r', label='max')
        plt.legend(loc="upper left")
        plt.xlabel('Epoch #')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms per Epoch\n{} optimizer with eta {}, {} cost function\nregularization {} with lambda {}'
                  .format(optimizer, self.network.cost, self.eta, self.network.regularization, self.network.reg_lambda))
        plt.grid()
        plt.show()
